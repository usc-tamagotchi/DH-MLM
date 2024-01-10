import argparse
import ipdb
import logging
import os
import re
import sys
import datasets
import torch
import json
import spacy
import pyinflect
from tqdm import tqdm
from nltk.tree import ParentedTree
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from train_glue import task_to_keys
from utils import get_model_name_from_path
from nltk.corpus import wordnet
import numpy as np
spacy_nlp = spacy.load('en_core_web_sm')


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
        splits = ['test', 'validation']

    has_pps = {}
    perturbeds = {}

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model
    )
    device = torch.device('cuda:0')
    model = AutoModelForSequenceClassification.from_pretrained(
        args.target_model
    ).to(device)

    rng = np.random.default_rng(args.seed)
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

                words = find_important_words(
                    model, tokenizer, device, tree
                )
                if words is None or len(words) == 0:
                    logging.warning('find no important words.')
                    continue

                # make a perturbed sample
                for word in words[:args.n_replace]:
                    perturbed = dict(sample)
                    synonym = get_synonym(word, rng)
                    if synonym is None:
                        continue

                    replaced = ' '.join(word.leaves())
                    if args.task == 'mnli':
                        if tree_col == 'hypothesis_parse':
                            perturbed['hypothesis'] = \
                                perturbed['hypothesis'].replace(
                                    replaced, synonym
                                )
                        else:
                            perturbed['premise'] = \
                                perturbed['premise'].replace(
                                    replaced, synonym
                                )
                    else:
                        perturbed[
                            task_to_keys[args.task][0]
                        ] = perturbed[
                            task_to_keys[args.task][0]
                        ].replace(replaced, synonym)

                    perturbed['replaced'] = replaced
                    perturbed['replacement'] = synonym
                    perturbed['tag'] = word.label()

                    perturbeds[split].append(
                        to_glue_format(perturbed, args.task, idx)
                    )

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
    path_save = os.path.join(dir_out, 'synonym', 'paraphrased.ds')
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


def find_important_words(model, tokenizer, device, tree,
                         batch_size=8):
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
    subtrees = [
        st for st in subtrees
        if len(st.leaves()) == 1
    ]

    if len(subtrees) == 0:
        return None

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

    sorted_subtrees = sorted(zip(prob_origs, subtrees))
    _, sorted_subtrees = zip(*sorted_subtrees)
    return sorted_subtrees


def get_synonym(word, rng):
    assert len(word.leaves()) == 1
    text = word.leaves()[0]
    lemma = spacy_nlp(text)[0].lemma_
    pos_tag = word.label()

    if pos_tag[0] == 'N':
        synsets = wordnet.synsets(text, pos=wordnet.NOUN)
    elif pos_tag[0] == 'V':
        synsets = wordnet.synsets(text, pos=wordnet.VERB)
    elif pos_tag[0] == 'J' or pos_tag[:3] == 'ADJ':
        synsets = wordnet.synsets(text, pos=wordnet.ADJ)
    else:
        raise NotImplementedError

    synonyms = [
        slemma.name()
        for synset in synsets
        for slemma in synset.lemmas()
        if slemma.name() != lemma
    ]
    if len(synonyms) == 0:
        return None

    synonym = synonyms[rng.integers(len(synonyms))]
    return spacy_nlp(synonym)[0]._.inflect(pos_tag)


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('task', type=str, help='')
    parser.add_argument('--target_model', type=str, default=None,
                        help='will find the shortest most impactful '
                        'phrase when this arg is specified')
    parser.add_argument('--n_replace', type=int, default=4, help='')
    parser.add_argument('--seed', type=int, default=329, help='')
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
