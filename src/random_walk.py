import argparse
import ipdb
import logging
import os
import sys
import re
import datasets
import torch
from tqdm import tqdm
import numpy as np


def main(args):
    dir_out = args.dir_out or os.path.join('..', 'data', 'random_walks')
    os.makedirs(dir_out, exist_ok=True)
    torch.save(args, os.path.join(dir_out, 'args.pt'))

    # generate unlabeled data
    random_walks = [RandomWalk(
        args.rw_n_node, args.vocab_size, args.rw_degree,
        args.seed + i
    ) for i in range(2)]
    unlabeled = {}
    for i in range(2):
        sizes = [
            args.n_unlabeled,
            args.n_unlabeled // 10,
            args.n_unlabeled // 10
        ]
        for split, size in zip(['train', 'valid', 'test'], sizes):
            logging.info(f'Generating unlabeled data for {split}-{i}')
            unlabeled[f'{split}-{i}'] = datasets.Dataset.from_dict({
                'input_ids': random_walks[i].sample(
                    size, args.unlabeled_seq_len
                )
            })

    unlabeled = datasets.DatasetDict(unlabeled)
    path_save = os.path.join(dir_out, 'unlabeled.ds')
    logging.info(f'Saving to {path_save}')
    unlabeled.save_to_disk(path_save)

    # generate labeled data
    nfa = NFA(
        args.vocab_size, args.n_nfa,
        args.nfa_n_node, args.nfa_n_edge, args.seed + 3
    )
    labeled = {}
    for i in range(2):
        sizes = [
            args.n_labeled // 10,
            args.n_labeled // 10,
            args.n_labeled,
        ]
        for split, size in zip(['valid', 'test', 'train'], sizes):
            xs = random_walks[i].sample(size, args.labeled_seq_len)
            ys = nfa.label(xs)
            labeled[f'{split}-{i}'] = datasets.Dataset.from_dict({
                'input_ids': xs, 'label': ys
            })
            logging.info(
                f'ratio of positive samples in {split} '
                f'({size}) = {ys.mean():.2f}'
            )
    labeled = datasets.DatasetDict(labeled)
    path_save = os.path.join(dir_out, 'labeled.ds')
    logging.info(f'Saving to {path_save}')
    labeled.save_to_disk(path_save)


class NFA:
    def __init__(self, vocab_size,  n_nfa, n_node, n_edge, seed=None):
        self.vocab_size = vocab_size
        self.n_node = n_node
        self.n_edge = n_edge
        self.rng = np.random.default_rng(seed)
        self.patterns = [
            self._sample_pattern(n_node, n_edge)
            for _ in range(n_nfa)
        ]

    def _sample_pattern(self, n_node, n_edge):
        pattern = '^([0-9]+ )*'
        nodes = []
        for _ in range(n_node):
            edges = self.rng.choice(self.vocab_size, size=n_edge, replace=False)
            nodes.append(
                '(' + '|'.join([f'{e}' for e in edges]) + ')' + '( [0-9]+)*'
            )

        pattern += ' '.join(nodes) + '( [0-9]+)*$'

        return re.compile(pattern)

    def _match(self, sample, pattern):
        sample = ' '.join(map(str, sample))
        return pattern.match(sample)

    def _label(self, sample):
        for pattern in self.patterns:
            if self._match(sample, pattern):
                return 1

        return 0

    def label(self, samples):
        labels = np.array([
            self._label(sample)
            for sample in tqdm(samples)
        ])
        return labels


class RandomWalk:
    def __init__(self, n_node, vocab_size, degree, seed):
        assert vocab_size < n_node
        self.vocab_size = vocab_size
        self.degree = degree

        self.rng = np.random.default_rng(seed)
        assert n_node % vocab_size == 0
        seq = np.arange(vocab_size)
        self.rng.shuffle(seq)
        self.outputs = np.concatenate([
            seq
            for _ in range(n_node // vocab_size)
        ])

        self.neighbors = []

        assert degree > 1
        self.pt = np.array(
            [0.9] +
            [0.1 / (degree - 1)] * (degree - 1)
        )  # probability of transition to neighbors

        for i in range(n_node):
            rns = self.rng.choice(
                vocab_size - 1, size=degree - 1, replace=False
            )
            neighbors = (rns + i) % n_node
            self.neighbors.append(neighbors)

        self.neighbors = np.stack(self.neighbors, 0)
        assert self.neighbors.shape == (n_node, degree - 1)

        self.neighbors = np.concatenate([
            ((np.arange(n_node) + 1) % n_node)[:, None],
            self.neighbors,
        ], 1)
        assert self.neighbors.shape == (n_node, degree)

    def sample(self, n_sample, seq_len, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        assert seq_len > 0

        ns = rng.integers(0, len(self.outputs), size=n_sample)
        outputs = [
            self.outputs[ns]
        ]
        for _ in range(seq_len - 1):
            neighbors = self.neighbors[ns]
            rand_idx = rng.choice(self.degree, size=n_sample, p=self.pt)
            ns = np.take_along_axis(
                neighbors, rand_idx.reshape(-1, 1), axis=1
            ).reshape(-1)
            assert ns.shape == (n_sample, )

            outputs.append(
                self.outputs[ns]
            )

        outputs = np.stack(outputs, -1)
        assert outputs.shape == (n_sample, seq_len)
        return outputs


def _parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--seed', type=int, default=330, help='')
    parser.add_argument('--dir_out', type=str, default=None, help='')
    parser.add_argument('--vocab_size', type=int, default=1000, help='')
    parser.add_argument('--rw_n_node', type=int, default=5000, help='')
    parser.add_argument('--rw_degree', type=int, default=2, help='')
    parser.add_argument('--nfa_n_node', type=int, default=3, help='')
    parser.add_argument('--nfa_n_edge', type=int, default=12, help='')
    parser.add_argument('--n_nfa', type=int, default=5, help='')
    parser.add_argument('--n_unlabeled', type=int, default=100000, help='')
    parser.add_argument('--unlabeled_seq_len', type=int, default=512, help='')
    parser.add_argument('--n_labeled', type=int, default=100000, help='')
    parser.add_argument('--labeled_seq_len', type=int, default=100, help='')
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
