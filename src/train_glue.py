#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers
# for this are left as comments.

import re
import logging
import sys
import os
import torch
import json
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import IntervalStrategy
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils import get_model_name_from_path


check_min_version("4.21.0")

require_version("datasets>=1.8.0", "")


task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "sst2": ("sentence", ),
}


@dataclass
class ExperientArguments:
    dataset: Optional[str] = field(
        default='mnli'
    )
    path_dataset: Optional[str] = field(
        default=None
    )
    max_seq_length: Optional[int] = field(
        default=128,
    )
    valid_split_ratio: Optional[float] = field(
        default=0.2,
    )
    n_epoch: Optional[int] = field(
        default=3,
    )
    eval_interval: Optional[int] = field(
        default=1000
    )
    reinit: bool = field(
        default=False
    )
    eval_shift: bool = field(default=False)
    train_ratio: Optional[float] = field(
        default=1.0
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default='bert-base-uncased'
    )
    tokenizer_name_or_path: Optional[str] = field(
        default='bert-base-uncased', metadata={"help": ""}
    )


@dataclass
class GlueTrainingArguments(TrainingArguments):
    output_dir: Optional[str] = field(
        default=None
    )
    save_strategy: Optional[str] = field(
        default='no'
    )
    eval_strategy: Optional[str] = field(
        default='epoch'
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, ExperientArguments, GlueTrainingArguments)
    )
    model_args, exp_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.output_dir is None:
        training_args.output_dir = os.path.join(
            '..', 'models',
            f'{exp_args.dataset}-{model_args.model_name_or_path}'
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = 'INFO'
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
    )
    num_labels = {'mnli': 3, 'sst2': 2}[exp_args.dataset]
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    if exp_args.reinit:
        model.init_weights()

    # Preprocessing the raw_datasets
    max_seq_length = min(exp_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_fn(examples):
        inputs = (
            examples[key]
            for key in task_to_keys[exp_args.dataset]
        )

        # Tokenize the texts
        result = tokenizer(*inputs, max_length=max_seq_length, truncation=True)
        return result

    if exp_args.path_dataset is None:
        ds = load_dataset('glue', exp_args.dataset)
    else:
        ds = datasets.load_from_disk(exp_args.path_dataset)
        logger.info(f'Using dataset from {exp_args.path_dataset} .')

    if exp_args.train_ratio < 1.0:
        n_total = len(ds['train'])
        n_subset = int(n_total * exp_args.train_ratio)
        rng = np.random.default_rng(training_args.seed)
        indices = rng.choice(n_total, size=n_subset, replace=False)
        ds['train'] = ds['train'].select(indices)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        ds = ds.map(
            preprocess_fn,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds.get('train', None),
        eval_dataset=ds.get('validation_matched', None),
        tokenizer=tokenizer,
        data_collator=None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

    accs = {}
    if training_args.do_predict:
        if exp_args.path_dataset is None:
            dir_out = training_args.output_dir
        else:
            path_dataset = exp_args.path_dataset
            if path_dataset[-1] == '\\':
                path_dataset = path_dataset[:-1]
            ds_dir = os.path.dirname(path_dataset)
            model_name = get_model_name_from_path(
                model_args.model_name_or_path, include_ckpt=True
            )
            dir_out = os.path.join(ds_dir, model_name)

        os.makedirs(dir_out, exist_ok=True)
        logger.info(f'Output directory: {dir_out}')
        splits = [k for k in ds.keys() if k != 'train']
        for split in splits:
            ds_split = ds[split]
            if split == 'test':
                def map_neg_class(x):
                    if x['label'] == -1:
                        x['label'] = -100
                    return x
                ds_split = ds_split.map(map_neg_class)
            ys_, acc = do_predict(trainer, ds_split)
            path_pred = os.path.join(dir_out, f'pred-{split}.pt')
            logger.info(f'Saving to {path_pred} ...')
            torch.save(
                ys_, path_pred
            )
            print(f'Accuracy on {split} = {acc:.2f}')
            accs[split] = acc

        if exp_args.eval_shift:
            def shift_id(examples):
                examples['input_ids'] = [
                    [tid + tokenizer.vocab_size for tid in tids]
                    for tids in examples['input_ids']
                ]
                return examples
            shifted = ds[split].map(shift_id, batched=True)
            ys_, acc = do_predict(trainer, shifted)
            path_pred = os.path.join(
                training_args.output_dir, f'pred-shift-{split}.pt'
            )
            torch.save(
                ys_, path_pred
            )
            print(f'Accuracy on {split} = {acc:.2f}')
            accs[f'{split}-shifted'] = acc

        path_json = os.path.join(
            dir_out, 'accs.json'
        )
        with open(path_json, 'w') as f:
            json.dump(
                accs, f
            )
        logger.info(f'Saved to {path_json}')


def do_predict(trainer, ds):
    preds = trainer.predict(ds).predictions
    ys_ = preds.argmax(axis=1)
    ys = np.array(ds['label'])
    acc = (ys_ == ys).mean()
    return preds, acc


def tokenize(text):
    return re.sub("[^a-z ']", "", text.lower()).strip().split()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
