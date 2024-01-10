import math
import ipdb
import logging
import os
import sys
import torch
import datasets
import transformers
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling,
    IntervalStrategy
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from torch.utils.tensorboard import SummaryWriter
from transformers.modeling_outputs import MaskedLMOutput
from gensim.models import Word2Vec


check_min_version('4.21.0')
require_version('datasets>=1.8.0')


logger = logging.getLogger(__name__)


@dataclass
class MLMTrainingArguments(TrainingArguments):
    evaluation_strategy: Optional[str] = field(
        default='epoch', metadata={'help': ''}
    )
    output_dir: Optional[str] = field(
        default=None, metadata={'help': ''}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=16, metadata={'help': ''}
    )
    per_device_eval_batch_size: Optional[str] = field(
        default=16, metadata={'help': ''}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={'help': ''}
    )
    save_total_limit: Optional[int] = field(
        default=2, metadata={'help': ''}
    )
    save_steps: Optional[int] = field(
        default=500, metadata={'help': ''}
    )
    logging_steps: Optional[int] = field(
        default=25, metadata={'help': ''}
    )
    eval_steps: Optional[int] = field(
        default=500, metadata={'help': ''}
    )
    seed: Optional[int] = field(
        default=329, metadata={'help': ''}
    )
    learning_rate: Optional[float] = field(
        default=5e-5, metadata={'help': ''}
    )
    weight_decay: Optional[float] = field(
        default=0.001, metadata={'help': ''}
    )
    adam_beta2: Optional[float] = field(
        default=0.98, metadata={'help': ''}
    )
    adam_epsilon: Optional[float] = field(
        default=1e-6, metadata={'help': ''}
    )
    warmup_ratio: Optional[float] = field(
        default=0.1, metadata={'help': ''}
    )
    num_train_epochs: Optional[int] = field(
        default=15, metadata={'help': ''}
    )


@dataclass
class ExpArguments:
    input_id_setting: Optional[str] = field(
        default=None, metadata={'help': ''}
    )
    model_name_or_path: Optional[str] = field(
        default=None, metadata={'help': ''}
    )
    level_up: Optional[bool] = field(
        default=False
    )
    resume: Optional[str] = field(
        default=None, metadata={'help': ''}
    )


def main():
    parser = HfArgumentParser((MLMTrainingArguments, ExpArguments))
    training_args, exp_args = parser.parse_args_into_dataclasses()

    assert exp_args.input_id_setting in [None, 'mix', 'shift', 'mix-shift']

    if training_args.output_dir is None:
        if exp_args.level_up:
            ds_name = 'level_up'
        else:
            ds_name = 'random_walks'

        if exp_args.input_id_setting is None:
            training_args.output_dir = os.path.join(
                '..', 'models', ds_name, 'glove'
            )
        else:
            training_args.output_dir = os.path.join(
                '..', 'models', ds_name,
                f'glove-{exp_args.input_id_setting}'
            )

    if training_args.do_train:
        os.makedirs(training_args.output_dir, exist_ok=True)
        path_args = os.path.join(training_args.output_dir, 'args.pt')
        torch.save([training_args, exp_args], path_args)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load dataset
    path_ds = os.path.join('..', 'data', ds_name, 'unlabeled.ds')
    logging.info(f'loading datasets from {path_ds}')
    dss = datasets.load_from_disk(path_ds)
    path_ds_arg = os.path.join('..', 'data', ds_name, 'args.pt')
    arg_ds = torch.load(path_ds_arg)

    def shift_id(examples):
        examples['input_ids'] = [
            [tid + arg_ds.vocab_size for tid in tids]
            for tids in examples['input_ids']
        ]
        return examples

    rng = np.random.default_rng(training_args.seed)

    def mix_id(examples):
        examples['input_ids'] = [
            [tid + arg_ds.vocab_size * int(rng.random() > 0.5)
             for tid in tids]
            for tids in examples['input_ids']
        ]
        return examples

    with training_args.main_process_first(desc='dataset map pre-processing'):
        if exp_args.input_id_setting == 'shift':
            for split in ['train', 'valid']:
                dss[f'{split}-1'] = dss[f'{split}-1'].map(
                    shift_id, batched=True
                )

        ds = datasets.DatasetDict({
            'train': datasets.concatenate_datasets(
                [dss['train-0'], dss['train-1']]
            ),
            'valid': datasets.concatenate_datasets(
                [dss['valid-0'], dss['valid-1']]
            )
        })
        if exp_args.input_id_setting == 'mix':
            if not exp_args.level_up:
                ds = ds.map(
                    mix_id, batched=True
                )
            else:
                ds = datasets.DatasetDict({
                    'train': datasets.concatenate_datasets(
                        [dss['train-0-mix'], dss['train-1-mix']]
                    ),
                    'valid': datasets.concatenate_datasets(
                        [dss['valid-0-mix'], dss['valid-1-mix']]
                    )
                })
        elif exp_args.input_id_setting == 'mix-shift':
            assert exp_args.level_up
            ds = datasets.DatasetDict({
                'train': datasets.concatenate_datasets(
                    [dss['train-0-mix-shift'], dss['train-1-mix-shift']]
                ),
                'valid': datasets.concatenate_datasets(
                    [dss['valid-0-mix-shift'], dss['valid-1-mix-shift']]
                )
            })

    if training_args.do_train:
        model = Word2Vec(
            sentences=ds['train']['input_ids'],
            vector_size=768, window=5, min_count=1, workers=16
        )
        model.train(
            ds['train']['input_ids'], epochs=10,
            total_examples=model.corpus_count
        )
        wvs = np.stack([
            model.wv[i]
            for i in range(len(model.wv))
        ], 1)
        path_save = os.path.join(training_args.output_dir, 'w2v.pt')
        logger.info(f'Saving to {path_save} ...')
        torch.save(wvs, path_save)


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        main()
