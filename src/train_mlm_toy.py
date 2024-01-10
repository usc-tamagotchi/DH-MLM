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
    max_span_len: Optional[int] = field(
        default=-1, metadata={'help': ''}
    )
    n_layer: Optional[int] = field(
        default=6, metadata={'help': ''}
    )
    mlm_probability: Optional[float] = field(
        default=0.15, metadata={'help': ''}
    )


def main():
    parser = HfArgumentParser((MLMTrainingArguments, ExpArguments))
    training_args, exp_args = parser.parse_args_into_dataclasses()

    assert exp_args.input_id_setting in [None, 'mix', 'shift', 'mix-shift']

    if exp_args.level_up:
        ds_name = 'level_up'
    else:
        ds_name = 'random_walks'

    # making output dir name
    if training_args.output_dir is None:
        prefix = ''
        if exp_args.max_span_len > 0:
            # assert exp_args.max_span_len == 3
            prefix = f'{prefix}-span'
        if exp_args.input_id_setting is not None:
            prefix = f'{prefix}-{exp_args.input_id_setting}'
        training_args.output_dir = os.path.join(
            '..', 'models', ds_name, f'mlm{prefix}'
        )

    if training_args.do_train:
        os.makedirs(training_args.output_dir, exist_ok=True)
        path_args = os.path.join(training_args.output_dir, 'args.pt')
        torch.save([training_args, exp_args], path_args)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'n_gpu: {training_args.n_gpu}, '
        f'distributed training: {bool(training_args.local_rank != -1)}, '
        f'16-bits training: {training_args.fp16}'
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load dataset
    path_ds = os.path.join('..', 'data', ds_name, 'unlabeled.ds')
    logging.info(f'loading datasets from {path_ds}')
    dss = datasets.load_from_disk(path_ds)
    path_ds_arg = os.path.join('..', 'data', ds_name, 'args.pt')
    arg_ds = torch.load(path_ds_arg)

    # set up model
    if exp_args.model_name_or_path is None:
        config = BertConfig.from_pretrained(
            'bert-base-cased',
            vocab_size=arg_ds.vocab_size * 2 + 1,  # +1 for <mask>
            num_hidden_layers=exp_args.n_layer,
            # initializer_range=math.sqrt(2 / (5 * 768)),
            num_attention_heads=12
        )
        model = BertForMaskedLM(config)
        model.init_weights()
        # model = CNN(arg_ds.vocab_size * 2 + 1)
    else:
        model = BertForMaskedLM.from_pretrained(exp_args.model_name_or_path)

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

    # this is just a placeholder for DataCollatorForLanguageModeling
    # tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    data_collator = DataCollatorForMLM(
        mask_token_id=arg_ds.vocab_size * 2,
        max_token_id=arg_ds.vocab_size * 2,
        mlm_probability=exp_args.mlm_probability,
        max_span_len=exp_args.max_span_len,
        rng=rng
    )

    # tokenizer.vocab_size = arg_ds.vocab_size * 2
    # tokenizer.mask_token_id = arg_ds.vocab_size * 2 + 1
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer, mlm=True, mlm_probability=0.15
    # )
    tb_writer = SummaryWriter(training_args.output_dir)
    callback_tb = transformers.integrations.TensorBoardCallback(tb_writer)
    callbacks = [callback_tb]

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ds['train'],
        eval_dataset=ds['valid'],
        callbacks=callbacks
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=exp_args.resume)
        trainer.save_model(training_args.output_dir)

    if training_args.do_eval:
        device = torch.device('cuda:0')
        for batch in trainer.get_eval_dataloader():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            breakpoint()


class CNN(torch.nn.Module):
    def __init__(self, vocab_size, dim_hidden=768):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, dim_hidden)
        self.cnn = torch.nn.Conv1d(dim_hidden, dim_hidden, 5, padding=2)
        self.out = torch.nn.Conv1d(dim_hidden, vocab_size, 1)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels):
        emb = self.embedding(input_ids).permute(0, 2, 1)
        # emb.shape == [bs, hidden, seq_len]
        z = self.cnn(emb)
        logits = self.out(z)
        loss = self.loss(logits, labels)
        return MaskedLMOutput(loss, logits.permute(0, 2, 1), z)


@dataclass
class DataCollatorForMLM:
    def __init__(self, mask_token_id: int, max_token_id: int, mlm_probability,
                 max_span_len=-1, rng=None):
        self.mask_token_id = mask_token_id
        self.max_token_id = max_token_id
        self.mlm_probability = mlm_probability
        if max_span_len > 0:
            self.span_probs = torch.tensor(
                [0.2 * 0.8**(i-1) for i in range(1, 4)]
            )
            self.span_probs /= self.span_probs.sum()
        else:
            self.span_probs = None

    def __call__(self, examples):
        batch = {
            "input_ids": torch.tensor([x['input_ids'] for x in examples])
        }
        batch["input_ids"], batch["labels"] = \
            self.mask_tokens(batch["input_ids"][:, :256])
        return batch

    def mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        if self.span_probs is None:
            probability_matrix = torch.full(labels.shape, self.mlm_probability)        
            masked_indices = torch.bernoulli(probability_matrix).bool()

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.mask_token_id

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(
                self.max_token_id, labels.shape, dtype=torch.long
            )
            inputs[indices_random] = random_words[indices_random]

            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        else:
            masked_indices = torch.full(labels.shape, 0).bool()
            for i in range(masked_indices.shape[0]):
                masking_lens = torch.multinomial(
                    self.span_probs, 200, replacement=True
                ) + 1
                start_poss = torch.multinomial(
                    torch.ones(inputs.shape[1]), 200, replacement=False
                )
                budget = inputs.shape[1] * self.mlm_probability
                total_masked_len, j = 0, 0
                while total_masked_len < budget and j < 200:
                    j += 1
                    total_masked_len += masking_lens[j]
                    masked_indices[i][start_poss[j]:start_poss[j] + masking_lens[j]] = 1
            inputs[masked_indices] = self.mask_token_id

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        return inputs, labels


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
