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

import transformers
from transformers import (
    AutoConfig,
    BertForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from torch.utils.tensorboard import SummaryWriter


check_min_version("4.21.0")

require_version("datasets>=1.8.0", "")


@dataclass
class ToyTrainingArguments(TrainingArguments):
    evaluation_strategy: Optional[str] = field(
        default='epoch', metadata={'help': ''}
    )
    save_strategy: Optional[str] = field(
        default='epoch', metadata={'help': ''}
    )
    load_best_model_at_end: Optional[bool] = field(
        default=True,
    )
    metric_for_best_model: Optional[str] = field(
        default='eval_accuracy'
    )
    save_total_limit: Optional[int] = field(
        default=1
    )


@dataclass
class ExperimentArguments:
    eval_shift: bool = field(default=False)
    load_model: Optional[str] = field(
        default=None, metadata={"help": ""}
    )
    train_ratio: Optional[float] = field(
        default=1.0, metadata={'help': ''}
    )
    lang1_ratio: Optional[float] = field(
        default=0, metadata={'help': ''}
    )
    shuffle_weight: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )
    level_up: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )
    share_alphabet: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )
    load_embedding: Optional[str] = field(
        default=None, metadata={"help": ""}
    )
    freeze_embedding: Optional[bool] = field(
        default=False, metadata={'help': ''}
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default='roberta-base'
    )
    tokenizer_name_or_path: Optional[str] = field(
        default='roberta-base', metadata={"help": ""}
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, ExperimentArguments, ToyTrainingArguments)
    )
    model_args, exp_args, training_args = parser.parse_args_into_dataclasses()
    training_args.eval_steps = 500
    training_args.logging_steps = 200
    training_args.warmup_ratio = 0.1

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

    if exp_args.load_model is None:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=2,
            vocab_size=2001,
            num_hidden_layers=6,
            problem_type='single_label_classification',
        )
        model = BertForSequenceClassification(
            config=config,
        )
        model.init_weights()
    else:
        model = BertForSequenceClassification.from_pretrained(
            exp_args.load_model
        )

    if exp_args.shuffle_weight:
        model = model.apply(shuffle_weight)

    if exp_args.load_embedding:
        emb = torch.load(exp_args.load_embedding).reshape(-1, 768)
        emb = emb / emb.std() * 0.03
        model.bert.embeddings.word_embeddings.weight.data[:emb.shape[0]] \
            = torch.tensor(emb)

    if exp_args.freeze_embedding:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

    if not exp_args.level_up:
        path_dataset = os.path.join('..', 'data', 'random_walks', 'labeled.ds')
    else:
        path_dataset = os.path.join('..', 'data', 'level_up', 'labeled.ds')

    ds = datasets.load_from_disk(path_dataset)

    ds_args = torch.load(
        os.path.join(path_dataset, '..', 'args.pt')
    )

    def shift_id(examples):
        examples['input_ids'] = [
            [tid + ds_args.vocab_size for tid in tids]
            for tids in examples['input_ids']
        ]
        return examples

    rng = np.random.default_rng(training_args.seed)
    if exp_args.train_ratio < 1.0 or exp_args.lang1_ratio > 0:
        n_total = len(ds['train-0'])
        n_subset = int(n_total * exp_args.train_ratio)

        if exp_args.lang1_ratio == 0:
            indices = rng.choice(n_total, size=n_subset, replace=False)
            train = ds['train-0'].select(indices)
            valid = ds['valid-0']
        else:
            # make the training dataset
            n0 = int(n_subset * (1 - exp_args.lang1_ratio))
            n1 = int(n_subset * exp_args.lang1_ratio)
            indices0 = rng.choice(n_total, size=n0, replace=False)
            indices1 = rng.choice(n_total, size=n1, replace=False)

            if exp_args.share_alphabet:
                assert exp_args.level_up
                train1 = ds['train-1']
            else:
                train1 = ds['train-1'].map(shift_id, batched=True)

            train = datasets.concatenate_datasets([
                ds['train-0'].select(indices0),
                train1.select(indices1),
            ])

            # make the valid dataset
            n_valid = len(ds['valid-0'])
            n0 = int(n_valid * (1 - exp_args.lang1_ratio))
            n1 = int(n_valid * exp_args.lang1_ratio)
            indices0 = rng.choice(n_valid, size=n0, replace=False)
            indices1 = rng.choice(n_valid, size=n1, replace=False)
            valid1 = ds['valid-1'].map(shift_id, batched=True)
            valid = datasets.concatenate_datasets([
                ds['valid-0'].select(indices0),
                valid1.select(indices1),
            ])
    else:
        train = ds['train-0']
        valid = ds['valid-0']

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    tb_writer = SummaryWriter(training_args.output_dir)
    callback_tb = transformers.integrations.TensorBoardCallback(tb_writer)
    callback_es = transformers.EarlyStoppingCallback(5)
    callbacks = [callback_es, callback_tb]
    accuracy = datasets.load_metric("accuracy")

    if not exp_args.level_up:
        data_collator = None
    else:
        def data_collator(xs):
            inputs = [x['input_ids'] for x in xs]
            lens = [len[s] for s in inputs]
            max_len = max(lens)
            attention_mask = torch.tensor([
                [1] * ll + [0] * (max_len - ll)
                for ll in lens
            ])
            pad_idx = ds_args.vocab_size
            padded = torch.tensor([
                s + [pad_idx] * (max_len - ll)
                for s, ll in zip(inputs, lens)
            ])
            labels = torch.tensor([
                x['label'] for x in xs
            ])
            return {
                'input_ids': padded,
                'attention_mask': attention_mask,
                'label': labels
            }

    def compute_metric(pred):
        preds = pred.predictions.argmax(-1)
        return accuracy.compute(predictions=preds, references=pred.label_ids)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=tokenizer,
        data_collator=None,
        callbacks=callbacks,
        compute_metrics=compute_metric
    )

    # Training
    if training_args.do_train:
        torch.save(
            (model_args, exp_args, training_args),
            os.path.join(training_args.output_dir, 'args.pt')
        )
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()

    accs = {}
    if training_args.do_predict:
        splits = ['valid-0', 'test-0', 'valid-1', 'test-1']
        if exp_args.level_up:
            splits = splits + [f'{split}-tr' for split in splits]

        for split in splits:
            if split.split('-')[0] == 'train':
                continue

            ys_, acc = do_predict(trainer, ds[split])
            torch.save(
                ys_, os.path.join(
                    training_args.output_dir, f'pred-{split}.pt'
                )
            )
            print(f'Accuracy on {split} = {acc:.2f}')
            accs[split] = acc

            if exp_args.eval_shift:
                shifted = ds[split].map(shift_id, batched=True)
                ys_, acc = do_predict(trainer, shifted)
                torch.save(
                    ys_, os.path.join(
                        training_args.output_dir, f'pred-{split}-shift.pt'
                    )
                )
                print(f'Accuracy on {split}-shift = {acc:.2f}')
                accs[f'{split}-shift'] = acc

    path_json = os.path.join(
        training_args.output_dir, 'accs.json'
    )
    with open(path_json, 'w') as f:
        json.dump(
            accs, f
        )


def do_predict(trainer, ds):
    preds = trainer.predict(ds).predictions
    ys_ = preds.argmax(axis=1)
    ys = np.array(ds['label'])
    acc = (ys_ == ys).mean()
    return ys_, acc


def tokenize(text):
    return re.sub("[^a-z ']", "", text.lower()).strip().split()


def shuffle_weight(module, seed=329):
    rng = np.random.default_rng(seed)

    def shuffle_matrix(matrix):
        orig_shape = matrix.shape
        m = matrix.reshape(-1)
        indices = np.arange(m.shape[0])
        rng.shuffle(indices)
        m = m[indices]
        return m.reshape(orig_shape)

    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        module.weight.data = shuffle_matrix(module.weight.data)
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data = shuffle_matrix(module.bias.data)
        module.weight.data = shuffle_matrix(module.weight.data)
    if isinstance(module, torch.nn.Linear) and module.bias is not None:
        module.bias.data = shuffle_matrix(module.bias.data)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
