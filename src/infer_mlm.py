import argparse
import ipdb
import logging
import os
import sys
import datasets
import torch
import re
from utils import get_ds_name_from_path, get_model_name_from_path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import trange, tqdm
from train_glue import task_to_keys


def main(args):
    if args.path_ds is not None:
        path_ds = args.path_ds
    else:
        if args.target_model is not None:
            target_model_name = get_model_name_from_path(args.target_model)
            ds_name = 'paraphrased.ds'
            if not args.synonym:
                path_ds = os.path.join(
                    '..', 'data', args.task, target_model_name, ds_name
                )
            else:
                path_ds = os.path.join(
                    '..', 'data', args.task, target_model_name,
                    'synonym', ds_name
                )
        else:
            path_ds = os.path.join(
                '..', 'data', args.task, 'paraphrased.ds'
            )

    dss = datasets.load_from_disk(path_ds)
    logging.info(f'Loaded dataset from {path_ds}')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device('cuda:0')
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path, output_hidden_states=True
    ).to(device)

    def save(mlm_mode, mlm_outputs):
        if args.synonym:
            mlm_mode = 'synonym'

        # setup output dir
        if args.target_model is not None:
            model_name = get_model_name_from_path(args.target_model)
            dir_out = os.path.join(
                '..', 'data', args.task,
                f'mlm-outputs-{args.model_name_or_path}', model_name, mlm_mode
            )
        else:
            dir_out = os.path.join(
                '..', 'data', args.task,
                f'mlm-outputs-{args.model_name_or_path}', mlm_mode
            )
        os.makedirs(dir_out, exist_ok=True)

        # saving
        path_out = os.path.join(
            dir_out, f'{get_ds_name_from_path(path_ds)}.pt'
        )
        torch.save(mlm_outputs, path_out)
        logging.info(f'Saved to {path_out}')

    if args.mlm_phrase:
        mlm_outputs = mlm_phrase(model, tokenizer, dss, device)
        save('phrase', mlm_outputs)

    if args.mlm_with:
        mlm_outputs = mlm_with(model, tokenizer, dss, args.task, device)
        save('with', mlm_outputs)

    if args.mlm_mean:
        mlm_outputs = mlm_mean(model, tokenizer, dss, args.task, device)
        save('mean', mlm_outputs)

    if args.mlm_task:
        mlm_outputs = mlm_task(model, tokenizer, dss, args.task, device)
        save('task', mlm_outputs)


def mlm_task(model, tokenizer, dss, task='mnli',
             device=torch.device('cuda:0')):
    fields = task_to_keys[task]

    # extract sentences
    sents = set()
    maskeds = []
    for split, ds in dss.items():
        for x in tqdm(ds):
            sent = ' [SEP] '.join([x[field] for field in fields])
            sents.add(sent)

    # convert a set to a list in a deterministic way
    sents = sorted(sents)

    # make masked sentences with templates
    maskeds = []
    for sent in sents:
        if task == 'mnli':
            p, h = sent.split(' [SEP] ')
            masked = f'{p} ? {tokenizer.mask_token} , {h}'
        elif task == 'sst2':
            masked = f'{sent} It was {tokenizer.mask_token} .'

        maskeds.append(masked)

    mlm_outputs = infer_mlm(model, tokenizer, sents, maskeds, device)
    return mlm_outputs


def mlm_mean(model, tokenizer, dss, task='mnli',
             device=torch.device('cuda:0')):
    fields = task_to_keys[task]

    # extract sentences
    sents = set()
    for split, ds in dss.items():
        for x in tqdm(ds):
            for field in fields:
                sent = x[field].strip()
                if sent != '':
                    sents.add(sent)

    # convert a set to a list in a deterministic way
    sents = sorted(sents)

    maskeds = []
    for sent in sents:
        masked = f'This sentence : "{sent}" means {tokenizer.mask_token}'
        maskeds.append(masked)

    mlm_outputs = infer_mlm(model, tokenizer, sents, maskeds, device)
    return mlm_outputs


def mlm_with(model, tokenizer, dss, task='mnli', device=torch.device('cuda:0')):
    fields = task_to_keys[task]

    # extract sentences
    sents = set()
    for split, ds in dss.items():
        for x in tqdm(ds):
            for field in fields:
                sent = x[field].strip()
                if sent != '':
                    sents.add(sent)

    # convert a set to a list in a deterministic way
    sents = sorted(sents)

    has_with = 0
    maskeds = []
    for sent in sents:
        sent = sent.strip()
        if 'with' in re.sub(r"[^a-z ]", '', sent.lower()).split(' '):
            has_with += 1

        if sent[-1] in '!?."\';':
            # insert [mask] before the last punctuation
            masked = re.sub(
                r" *([!\.\"'\?;]+)$",
                f" with {tokenizer.mask_token} \\1",
                sent
            )
        else:
            masked = f'{sent} with {tokenizer.mask_token}'

        assert tokenizer.mask_token in masked
        maskeds.append(masked)

    logging.info(f'There are {has_with} sentence containing "with".')

    mlm_outputs = infer_mlm(model, tokenizer, sents, maskeds, device)
    return mlm_outputs


def mlm_phrase(model, tokenizer, dss, device=torch.device('cuda:0')):
    # extract phrases
    phrases = []
    tags = []
    for split, ds in dss.items():
        for x in tqdm(ds):
            if x['replacement'] != '':
                phrases.append(x['replacement'])
                tags.append(x['tag'])
            if x['replaced'] != '':
                phrases.append(x['replaced'])
                tags.append(x['tag'])

    masked = [
        add_mask_by_tag(phrase, tag, tokenizer.mask_token)
        for phrase, tag in zip(phrases, tags)
    ]

    mlm_outputs = infer_mlm(model, tokenizer, phrases, masked, device)
    return mlm_outputs


def infer_mlm(model, tokenizer, orig_sents, maskeds, device):
    mlm_outputs = {}
    with torch.no_grad():
        for start in trange(0, len(maskeds), args.batch_size):
            end = start + args.batch_size
            batch = {
                k: v.to(device)
                for k, v in tokenizer(
                        maskeds[start:end], return_tensors='pt', padding=True
                ).items()
            }
            outputs = model(**batch)

            for i in range(len(batch['input_ids'])):
                index_mask = batch['input_ids'][i].tolist().index(
                    tokenizer.mask_token_id
                )
                prob = outputs.logits[i, index_mask]
                z = outputs.hidden_states[0][i, index_mask]
                sent = orig_sents[start + i]
                mlm_outputs[sent] = {
                    'prob': prob.cpu(),
                    'hidden': z.cpu()
                }

    return mlm_outputs


def add_mask_by_tag(phrase, tag, mask_token):
    tag = tag.replace('[', '').replace(']', '')
    if tag in ['VP'] or tag[0] == 'V':
        return f'{mask_token} {phrase}'
    if tag in ['NP'] or tag[0] == 'N':
        return f'{phrase} {mask_token}'
    if tag in ['ADJP'] or tag[0] == 'J':
        return f'{mask_token} is {phrase}'

    raise NotImplementedError


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'task', type=str, help=''
    )
    parser.add_argument(
        '--path_ds',
        type=str, help='', default=None
    )
    parser.add_argument(
        '--batch_size', type=int, help='', default=8
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str, help='', default='bert-base-uncased'
    )
    parser.add_argument(
        '--target_model',
        type=str, help='', default=None
    )
    parser.add_argument(
        '--mlm_phrase', action='store_true',
        help='Infer MLM at the phrase level.'
    )
    parser.add_argument(
        '--mlm_with', action='store_true',
        help='Infer MLM using the template "[X] with [mask]."'
    )
    parser.add_argument(
        '--mlm_mean', action='store_true',
        help='Infer MLM using the template "[X] means [mask].'
    )
    parser.add_argument(
        '--mlm_task', action='store_true',
        help='Infer MLM using a task specific template.'
    )
    parser.add_argument(
        '--synonym', action='store_true',
        help=''
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        main(args)
