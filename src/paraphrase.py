import argparse
import ipdb
import logging
import os
import re
import sys
import datasets
import torch
import json
from tqdm import tqdm
from nltk.tree import ParentedTree
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
from train_glue import task_to_keys
from editdistance import distance as compute_edit_distance
from utils import get_model_name_from_path


def main(args):
    if args.task == 'mnli':
        ds = datasets.load_dataset('multi_nli')
        splits = ['validation_matched', 'validation_mismatched']
        tree_cols = ['premise_parse', 'hypothesis_parse']
    else:
        sst2 = datasets.load_dataset('glue', 'sst2')

        # reading parses from json
        with open('../data/parses/sst2/dev.txt.json') as f:
            # skip the first line because it's the header
            trees = extract_trees_from_json(json.load(f))[1:]
            dev = sst2['validation'].add_column('tree', trees)

        with open('../data/parses/sst2/test.txt.json') as f:
            trees = extract_trees_from_json(json.load(f))[1:]
            test = sst2['test'].add_column('tree', trees)

        ds = datasets.DatasetDict({
            'validation': dev,
            'test': test
        })
        tree_cols = ['tree']
        splits = ['validation', 'test']

    paraphraser = Paraphraser(args.model_name, num_return_seq=5)

    has_pps = {}
    perturbeds = {}

    if args.target_model is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model
        )
        device = torch.device('cuda:0')
        model = AutoModelForSequenceClassification.from_pretrained(
            args.target_model
        ).to(device)

    for split in splits:
        logging.info(f'Iterating through {split}')
        has_pps[split] = []
        perturbeds[split] = []
        for idx, sample in enumerate(tqdm(ds[split])):
            for tree_col in tree_cols:
                perturbeds[split].append(
                    to_glue_format(sample, args.task, idx)
                )

                tree = ParentedTree.fromstring(sample[tree_col])

                # attach index to terminal nodes
                terminals = tree.subtrees(lambda n: n.height() == 2)
                for i, t in enumerate(terminals):
                    t.idx = i

                subtrees = [
                    st for st in tree.subtrees()
                    if (st.label() in ['NP', 'VP', 'ADJP']
                        and st.height() < tree.height())
                ]

                if len(subtrees) == 0:
                    logging.info(f'No qualified subtrees found in {tree}')
                    continue

                if args.target_model is None:
                    # find the longest subtree
                    subtree = max((len(st.leaves()), st) for st in subtrees)[1]
                else:
                    subtree = find_important_subtree(
                        model, tokenizer, device, tree,
                        max_phrase_len=args.max_phrase_len
                    )

                    if subtree is None:
                        logging.info(f'Find no qualified subtree in {tree}.')
                        continue

                indices = [
                    st.idx
                    for st in subtree.subtrees(lambda n: n.height() == 2)
                ]
                idx_start, idx_end = indices[0], indices[-1] + 1
                prefix = ' '.join(tree.leaves()[:idx_start])
                content = ' '.join(tree.leaves()[idx_start:idx_end])
                suffix = ' '.join(tree.leaves()[idx_end:])

                # generate paraphrase for the phrase
                if subtree.label() == 'NP' and subtree.left_sibling() is None:
                    replacement = paraphraser.gen([content])[0].strip()
                else:
                    paraphrase = paraphraser.gen(
                        [prefix + ' ' + content], conds=[prefix]
                    )[0]
                    replacement = paraphrase[len(prefix):].strip()

                # make a perturbed sample
                perturbed = dict(sample)
                if args.task == 'mnli':
                    if tree_col == 'hypothesis_parse':
                        perturbed['hypothesis'] = \
                            f'{prefix} {replacement} {suffix}'
                    else:
                        perturbed['premise'] = \
                            f'{prefix} {replacement} {suffix}'
                else:
                    perturbed[
                        task_to_keys[args.task][0]
                    ] = f'{prefix} {replacement} {suffix}'

                perturbed['replaced'] = content
                perturbed['replacement'] = replacement
                perturbed['tag'] = subtree.label()

                perturbeds[split].append(
                    to_glue_format(perturbed, args.task, idx)
                )

    if args.target_model is None:
        dir_out = os.path.join('..', 'data', args.task)
    else:
        model_name = get_model_name_from_path(args.target_model)
        dir_out = os.path.join('..', 'data', args.task, model_name)

    os.makedirs(dir_out, exist_ok=True)

    perturbeds = {
        k:  {f: [x[f] for x in samples] for f in samples[0]}
        for k, samples in perturbeds.items()
    }
    ds_perturbed = datasets.DatasetDict(
        {k: datasets.Dataset.from_dict(v) for k, v in perturbeds.items()}
    )
    path_save = os.path.join(dir_out, 'paraphrased.ds')
    ds_perturbed.save_to_disk(path_save)


def to_glue_format(sample, task, idx):
    if task in ['mnli', 'mnli-small']:
        return {
            'orig_idx': idx,
            'label': sample['label'],
            'premise': sample['premise'],
            'hypothesis': sample['hypothesis'],
            'replaced': sample.get('replaced', ''),
            'replacement': sample.get('replacement', ''),
            'tag': sample.get('tag', '')
        }
    elif task == 'sst2':
        return {
            'orig_idx': idx,
            'label': sample['label'],
            'sentence': sample['sentence'],
            'replaced': sample.get('replaced', ''),
            'replacement': sample.get('replacement', ''),
            'tag': sample.get('tag', '')
        }


def extract_trees_from_json(parse):
    trees = []
    for sent in parse['sentences']:
        tree = re.sub('\n *', '', sent['parse'])
        trees.append(tree)
    return trees


def find_important_subtree(model, tokenizer, device, tree,
                           batch_size=8, max_phrase_len=-1):
    blacklist = ['he', 'him', 'she', 'her', 'it', 'someone', 'somebody']
    # make candidates
    subtrees = [
        st for st in tree.subtrees()
        if (
                (st.label() in ['NP', 'VP', 'ADJP']
                 or st.label()[0] in ['N', 'J', 'V'])
                and st.height() < tree.height()
                and ' '.join(st.leaves()).lower() not in blacklist
        )
    ]
    if len(subtrees) == 0:
        return None

    filtered_subtrees = subtrees
    if max_phrase_len > 0:
        subtrees = [
            st for st in subtrees
            if len(st.leaves()) <= max_phrase_len
        ]
    assert len(subtrees) > 0

    subtrees = sorted(subtrees, key=lambda st: len(st.leaves()))

    candidates = [' '.join(tree.leaves())]
    for subtree in subtrees:
        indices = [
            st.idx
            for st in subtree.subtrees(lambda n: n.height() == 2)
        ]
        idx_start, idx_end = indices[0], indices[-1] + 1

        prefix = ' '.join(tree.leaves()[:idx_start])
        suffix = ' '.join(tree.leaves()[idx_end:])
        candidates.append(f'{prefix} {tokenizer.mask_token} {suffix}')

    preds = []
    prob_origs = []
    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            tokenized = tokenizer(batch, return_tensors='pt', padding=True)
            probs = model(**tokenized.to(device)).logits.cpu().softmax(-1)
            preds += probs.argmax(-1).cpu().tolist()
            prob_origs += probs[:, preds[0]].cpu().tolist()

    subtree_lens = [len(subtree) for subtree in subtrees]
    sorted_subtrees = sorted(zip(subtree_lens, prob_origs, preds, subtrees))

    for _, prob_orig, pred, subtree in sorted_subtrees:
        n_label = probs.shape[1]
        thd = (1 / n_label) + 0.05
        if pred != preds[0] or prob_orig < thd:
            return subtree

    sorted_subtrees = sorted(zip(prob_origs, subtrees))
    _, min_subtree = sorted_subtrees[0]
    return min_subtree


class Paraphraser:
    def __init__(self, model_name, num_return_seq=1):
        self.device = torch.device('cuda:0')
        e2x = 'facebook/wmt19-en-de'
        x2e = 'facebook/wmt19-de-en'
        self.e2x = AutoModelForSeq2SeqLM.from_pretrained(e2x).to(self.device)
        self.x2e = AutoModelForSeq2SeqLM.from_pretrained(x2e).to(self.device)
        self.tokenizer_e2x = AutoTokenizer.from_pretrained(e2x)
        self.tokenizer_x2e = AutoTokenizer.from_pretrained(x2e)
        self.beam_size = num_return_seq
        self.num_return_seq = num_return_seq

    def gen(self, texts, conds=None):
        # translate from En to X
        batch = self.tokenizer_e2x(texts, return_tensors='pt')
        gen_ids = self.e2x.generate(
            batch['input_ids'].to(self.device),
            num_beams=self.beam_size
        )
        intermediate = self.tokenizer_e2x.batch_decode(
            gen_ids, skip_special_tokens=True
        )

        # prepare condition for generation
        if conds is None:
            forced_decoder_ids = None
        else:
            assert len(texts) == 1
            forced_decoder_ids = [
                (i + 1, id)
                for i, id in enumerate(
                    self.tokenizer_x2e(conds)['input_ids'][0][:-1]
                )
            ]

        # translate from X to En
        inputs = self.tokenizer_x2e(intermediate, return_tensors='pt')
        outputs = self.x2e.generate(
            inputs['input_ids'].to(self.device),
            forced_decoder_ids=forced_decoder_ids,
            num_beams=self.beam_size,
            num_return_sequences=self.num_return_seq,
            output_scores=True,
            return_dict_in_generate=True
        )
        gen_idss = outputs.sequences.cpu().tolist()
        gen_sents = self.tokenizer_x2e.batch_decode(
            gen_idss, skip_special_tokens=True
        )
        gen_scores = outputs.sequences_scores.cpu().tolist()
        gen_sents, gen_idss = zip(*[
            (sent, idss)
            for sent, idss, score in zip(gen_sents, gen_idss, gen_scores)
            if sent != ''
        ])
        gen_sents, gen_idss = zip(*[
            (sent, idss)
            for sent, idss, score in zip(gen_sents, gen_idss, gen_scores)
            if max(gen_scores) - score < 1
        ])

        max_edit_distance = 0
        max_diff_sent = 0
        if self.num_return_seq == 1:
            return gen_sents
        else:
            assert len(texts) == 1
            input_ids = self.tokenizer_x2e.encode(
                texts[0], add_special_tokens=False
            )
            for i, gids in enumerate(gen_idss):
                try:
                    gen_len = gids.index(self.tokenizer_x2e.pad_token_id)
                except ValueError:
                    gen_len = len(gids)

                ed = compute_edit_distance(gids[:gen_len], input_ids)
                if ed > max_edit_distance:
                    max_edit_distance = ed
                    max_diff_sent = gen_sents[i]

            if texts[0][-1] != '.' and max_diff_sent[-1] == '.':
                max_diff_sent = max_diff_sent[:-1]
            max_diff_sent = max_diff_sent.replace(" 't", "'t")

            return [max_diff_sent]


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('task', type=str, help='')
    parser.add_argument('--model_name', type=str,
                        default='eugenesiow/bart-paraphrase', help='')
    parser.add_argument('--target_model', type=str, default=None,
                        help='will find the shortest most impactful '
                        'phrase when this arg is specified')
    parser.add_argument('--max_phrase_len', type=int, default=4,
                        help='')
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
