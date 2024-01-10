import argparse
import ipdb
import logging
import os
import scipy
import sys
import torch
import datasets
import json
import plotly.express as px
import numpy as np
from train_glue import task_to_keys
import plotly.graph_objects as go
import plotly.colors
from transformers import AutoTokenizer
from collections import defaultdict
import plotly.graph_objects as go


def load_data(
        task, min_phrase=False, model='bert-base-uncased',
        ckpt=None, ratio=None, ds=None, synonym=False
):
    # mlm outputs
    if not min_phrase and not synonym:
        # paraphrased dataset
        if ds is None:
            ds = datasets.load_from_disk(f'../data/{task}/paraphrased.ds/')

        splits = [split for split in ds.keys() if split != 'train']

        # predictions
        if ckpt is None:
            dir_pred = f'../data/{task}/{task}-{model}/'
        else:
            dir_pred = f'../data/{task}/{task}-{model}-steps-ckpt-{ckpt}/'

        if ratio is not None:
            dir_pred = f'../data/{task}/{task}-{model}-{ratio}/'

        logging.info(f'Loading predctions from {dir_pred} ...')
        preds = {
            split: torch.load(f'{dir_pred}/pred-{split}.pt')
            for split in splits
        }

        dir_mlm_output = f'../data/{task}/mlm-outputs-{model}/'
        logging.info(f'Loading from {dir_mlm_output} ...')
        mlm_out = {
            'phrase': torch.load(f'{dir_mlm_output}/phrase/paraphrased.pt'),
            'with': torch.load(f'{dir_mlm_output}/with/paraphrased.pt'),
            'mean': torch.load(f'{dir_mlm_output}/mean/paraphrased.pt'),
            'task': torch.load(f'{dir_mlm_output}/task/paraphrased.pt'),
        }
        tokenizer = AutoTokenizer.from_pretrained(model)
        if task == 'mnli':
            verbalizers = ['yes', 'maybe', 'no']
        elif task == 'sst2':
            verbalizers = ['great', 'terrible']
        idx_verbalizers = [
            tokenizer.vocab[v] for v in verbalizers
        ]
        for sent, outputs in mlm_out['task'].items():
            outputs['prob'] = outputs['prob'][idx_verbalizers]
    else:
        # paraphrased dataset
        dir_ds = f'../data/{task}/{task}-{model}/'
        if synonym:
            dir_ds = os.path.join(dir_ds, 'synonym')
        if ds is None:
            ds = datasets.load_from_disk(os.path.join(dir_ds, 'paraphrased.ds'))
        splits = [split for split in ds.keys() if split != 'train']

        # predictions
        if ckpt is not None:
            dir_pred = os.path.join(dir_ds, f'{task}-{model}-steps-ckpt-{ckpt}')
        elif ratio is not None:
            dir_pred = os.path.join(dir_ds, f'{task}-{model}-{ratio}')
        else:
            dir_pred = os.path.join(dir_ds, f'{task}-{model}')

        logging.info(f'Loading predctions from {dir_pred} ...')
        preds = {
            split: torch.load(os.path.join(dir_pred, f'pred-{split}.pt'))
            for split in splits
        }

        dir_mlm_output = f'../data/{task}/mlm-outputs-{model}/'
        if synonym:
            mlm_out = {
                'phrase': torch.load(
                    f'{dir_mlm_output}/{task}-{model}/synonym/paraphrased.pt'
                ),
            }
        else:
            mlm_out = {
                'phrase': torch.load(
                    f'{dir_mlm_output}/{task}-{model}/phrase/paraphrased.pt'
                ),
            }

    return ds, preds, mlm_out


def make_ds_visualize(ds, preds, mlm_out, task):
    ds_pred = ds.copy()

    # mapping to the original prediction and inputs
    splits = [split for split in ds.keys() if split != 'train']
    for split in splits:
        orig_preds = {}
        orig_inputs = {}

        for pred, sample in zip(preds[split], ds[split]):
            pred = scipy.special.softmax(pred, -1)
            if sample['replaced'] == '':
                orig_preds[sample['orig_idx']] = pred
                orig_inputs[sample['orig_idx']] = ' [SEP] '.join([
                    sample[key] for key in task_to_keys[task]
                ])

        ds_pred[split] = ds_pred[split].add_column(
            'pred',
            list(scipy.special.softmax(preds[split], -1))
        )
        ds_pred[split] = ds_pred[split].add_column(
            'orig_pred',
            [orig_preds[x['orig_idx']] for x in ds[split]]
        )
        ds_pred[split] = ds_pred[split].add_column(
            'orig_input',
            [orig_inputs[x['orig_idx']] for x in ds[split]]
        )
        ds_pred[split] = ds_pred[split].filter(
            lambda x: x['replacement'].lower() != x['replaced'].lower()
            or x['replaced'] == ''
        )

    ds_pred = datasets.DatasetDict(ds_pred)

    def gather_diff(sample, mlm_out, task='mnli'):
        """ Compute jsd and l1d for mlm-with and mlm-mean """

        if len(task_to_keys[task]) > 1:
            orig_inputs = sample['orig_input'].split(' [SEP] ')
            for i, key in enumerate(task_to_keys[task]):
                replacement = sample['replacement']
                replaced = sample['replaced'].replace(" n't", "n't")
                if replacement in sample[key]: # and replaced in orig_inputs[i]:
                    sent1 = sample[key].strip()
                    sent2 = orig_inputs[i].strip()
        else:
            sent1 = sample[task_to_keys[task][0]].strip()
            sent2 = sample['orig_input'].strip()

        mlm_p1 = mlm_out[sent1]['prob'].softmax(-1).numpy()
        mlm_p2 = mlm_out[sent2]['prob'].softmax(-1).numpy()
        dl1 = l1d(mlm_p1, mlm_p2)
        djs = jsd(mlm_p1, mlm_p2)

        return dl1, djs

    def compute_diff(sample):
        """ Compute jsd and l1d for all probs """
        if sample['replaced'] == '' or sample['replacement'] == '':
            sample['l1d_pred'] = 0
            sample['jsd_pred'] = 0
            for mode in mlm_out.keys():
                sample[f'l1d_mlm_{mode}'] = 0
                sample[f'jsd_mlm_{mode}'] = 0
        else:
            pred = np.array(sample['pred'])
            orig_pred = np.array(sample['orig_pred'])
            sample['l1d_pred'] = l1d(orig_pred, pred)
            sample['jsd_pred'] = jsd(orig_pred, pred)

            # compute diff for phrase level mlm
            mlm_p1 = mlm_out['phrase'][sample['replaced']]['prob'].softmax(
                -1).numpy()
            mlm_p2 = mlm_out['phrase'][sample['replacement']]['prob'].softmax(
                -1).numpy()
            sample['l1d_mlm_phrase'] = l1d(mlm_p1, mlm_p2)
            sample['jsd_mlm_phrase'] = jsd(mlm_p1, mlm_p2)

            # # compute diff for mlm-with and mlm-mean
            for mlm_mode in ['with', 'mean']:
                if mlm_mode not in mlm_out:
                    continue

                sample[f'l1d_mlm_{mlm_mode}'], sample[f'jsd_mlm_{mlm_mode}']\
                    = gather_diff(
                        sample, mlm_out[mlm_mode], task
                    )

            if 'task' in mlm_out:
                sent1 = ' [SEP] '.join([
                    sample[key] for key in task_to_keys[task]
                ])
                sent2 = sample['orig_input']
                mlm_p1 = mlm_out['task'][sent1]['prob'].softmax(-1).numpy()
                mlm_p2 = mlm_out['task'][sent2]['prob'].softmax(-1).numpy()
                sample['l1d_mlm_task'] = l1d(mlm_p1, mlm_p2)
                sample['jsd_mlm_task'] = jsd(mlm_p1, mlm_p2)

        return sample

    logging.info('Start to compute diff ...')
    ds_diff = datasets.DatasetDict({
        k: v.map(
            compute_diff, load_from_cache_file=False,
            keep_in_memory=True,
            new_fingerprint='placeholderr'
        )
        for k, v in ds_pred.items()
    })

    def is_paraphrased(x):
        return x['replacement'] != '' and x['replaced'] != ''

    def add_color_tag(x):
        x['color_tag'] = f"{x['tag']}"
        return x

    ds_visualize = ds_diff.filter(
        is_paraphrased
    ).map(
        add_color_tag
    )
    ds_visualize = datasets.concatenate_datasets(
        [ds_visualize[split] for split in splits]
    )
    return ds_visualize


def kld(p, q):
    return np.sum(
        p * (np.log(p) - np.log(q))
    )


def jsd(p, q):
    m = (p + q) / 2
    return kld(p, m) / 2 + kld(q, m) / 2


def l1d(p, q):
    return np.sum(np.abs(p - q))


def plot_accss(accss, suffixes, mode=None):
    colors = plotly.colors.qualitative.Plotly

    xs = [i / 10 for i in range(1, 11)]
    figure = go.Figure()

    if mode is None:
        keys = ['test-0', 'test-0-shift', 'test-1', 'test-1-shift']
    elif mode == 'level-up':
        keys = ['test-0', 'test-0-tr', 'test-1-tr', 'test-1']
    elif mode == 'level-up-shift':
        keys = ['test-0', 'test-0-tr-shift', 'test-1-tr', 'test-1-shift']

    names = ['D0 L0', 'D0 L1', 'D1 L0', 'D1 L1']
    dashes = ['solid', 'dash', 'dashdot', 'dot']

    for color, key, name in zip(colors, keys, names):
        for accs, suffix, dash in zip(accss, suffixes, dashes):
            figure.add_trace(
                go.Scatter(
                    x=xs, y=accs[key], name=f'{name} {suffix}',
                    line=dict(color=color, width=2, dash=dash)
                )
            )

    return figure


def main(args):
    corr_ckpts = {}
    corr_ratios = {}
    task_ckpts = {}
    for task in ['sst2', 'mnli']:
        ckpts = [
            int(name.replace('checkpoint-', ''))
            for name in os.listdir(
                    f'../models/{task}-bert-base-uncased-steps/'
            )
            if 'checkpoint-' in name
        ]
        task_ckpts[task] = ckpts
        corr_ckpts[task] = defaultdict(dict)
        corr_ratios[task] = defaultdict(list)

        ds = None
        for ckpt in ckpts:
            ds, preds, mlm_out = load_data(task, min_phrase=False,
                                           ckpt=ckpt, ds=ds)
            ds_visualize = make_ds_visualize(ds, preds, mlm_out, task)
            dir_out = os.path.join('..', 'outputs', task, f'ckpt-{ckpt}')
            os.makedirs(dir_out, exist_ok=True)
            for mlm_mode in ['phrase', 'task', 'mean', 'with']:
                corr = make_outputs(ds_visualize, dir_out, mlm_mode, mlm_mode)
                corr_ckpts[task][mlm_mode][ckpt] = corr

        for ratio in ['0.1', '0.2', '0.3', '0.4', '0.5']:
            ds, preds, mlm_out = load_data(task, min_phrase=False,
                                           ratio=ratio, ds=ds)
            ds_visualize = make_ds_visualize(ds, preds, mlm_out, task)
            dir_out = os.path.join('..', 'outputs', task, f'ratio-{ratio}')
            os.makedirs(dir_out, exist_ok=True)
            for mlm_mode in ['phrase', 'task', 'mean', 'with']:
                corr = make_outputs(ds_visualize, dir_out, mlm_mode, mlm_mode)
                corr_ratios[task][mlm_mode].append(corr)

        # make outputs for min phrase
        ds = None
        for ckpt in ckpts:
            ds, preds, mlm_out = load_data(task, min_phrase=True,
                                           ckpt=ckpt, ds=ds)
            ds_visualize = make_ds_visualize(ds, preds, mlm_out, task)
            dir_out = os.path.join('..', 'outputs', task, f'ckpt-{ckpt}')
            for mlm_mode in ['phrase']:
                corr = make_outputs(
                    ds_visualize, dir_out, mlm_mode, 'min-phrase'
                )
                corr_ckpts[task]['min-phrase'][ckpt] = corr

        ratios = ['0.1', '0.2', '0.3', '0.4', '0.5']
        for ratio in ratios:
            ds, preds, mlm_out = load_data(task, min_phrase=True,
                                           ratio=ratio, ds=ds)
            ds_visualize = make_ds_visualize(ds, preds, mlm_out, task)
            dir_out = os.path.join('..', 'outputs', task, f'ratio-{ratio}')
            os.makedirs(dir_out, exist_ok=True)
            for mlm_mode in ['phrase']:
                corr = make_outputs(
                    ds_visualize, dir_out, mlm_mode, 'min-phrase'
                )
                corr_ratios[task]['min-phrase'].append(corr)

        # make outputs for synonym
        ds = None
        ratios = ['0.1', '0.2', '0.3', '0.4', '0.5']
        for ratio in ratios:
            ds, preds, mlm_out = load_data(task, synonym=True,
                                           ratio=ratio, ds=ds)
            ds_visualize = make_ds_visualize(ds, preds, mlm_out, task)
            dir_out = os.path.join('..', 'outputs', task, f'ratio-{ratio}')
            os.makedirs(dir_out, exist_ok=True)
            for mlm_mode in ['phrase']:
                corr = make_outputs(ds_visualize, dir_out, mlm_mode, 'synonym')
                corr_ratios[task]['synonym'].append(corr)

        for ckpt in ckpts:
            ds, preds, mlm_out = load_data(task, synonym=True,
                                           ckpt=ckpt, ds=ds)
            ds_visualize = make_ds_visualize(ds, preds, mlm_out, task)
            dir_out = os.path.join('..', 'outputs', task, f'ckpt-{ckpt}')
            os.makedirs(dir_out, exist_ok=True)
            for mlm_mode in ['phrase']:
                corr = make_outputs(
                    ds_visualize, dir_out, mlm_mode, 'synonym'
                )
                corr_ckpts[task]['synonym'][ckpt] = corr

        figure = go.Figure()
        for mlm_mode in corr_ckpts[task]:
            ckpts = sorted(corr_ckpts[task][mlm_mode].keys())
            corrs = [corr_ckpts[task][mlm_mode][k] for k in ckpts]
            figure.add_trace(
                go.Scatter(x=ckpts, y=corrs, name=mlm_mode)
            )
        path_figure = os.path.join('..', 'outputs', task, 'ckpt.pdf')
        figure.write_image(path_figure)

        figure = go.Figure()
        for mlm_mode in corr_ratios[task]:
            figure.add_trace(
                go.Scatter(x=ratios, y=corr_ratios[task][mlm_mode],
                           name=mlm_mode)
            )
        path_figure = os.path.join('..', 'outputs', task, 'ratio.pdf')
        figure.write_image(path_figure)


def make_outputs(ds_visualize, dir_out, mlm_mode, save_name):
    n = min(3000, len(ds_visualize))
    subset = ds_visualize.shuffle(seed=329).select(list(range(n)))
    keyx = 'l1d_pred'

    keyy = f'l1d_mlm_{mlm_mode}'

    raw_stats = ds_visualize[keyx], ds_visualize[keyy]
    corr, p = scipy.stats.pearsonr(*raw_stats)

    path_raw = os.path.join(dir_out, f'{save_name}.pt')
    torch.save(raw_stats, path_raw)
    path_stats = os.path.join(dir_out, f'{save_name}.json')
    with open(path_stats, 'w') as f:
        json.dump({'corr': corr, 'p': p}, f)

    fig = px.scatter(
        subset,
        x=keyx,
        y=keyy,
        color='color_tag',
        hover_data=['replaced', 'replacement'],
        title=f'Pearson r = {corr:.2f}'
    )

    path_img = os.path.join(dir_out, f'{save_name}.pdf')
    logging.info(f'Saving image to {path_img} ...')
    fig.write_image(path_img)
    return corr


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument('arg1', type=None, help='')
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
