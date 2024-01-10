# The Distributional Hypothesis Does Not Fully Explain the Benefits of Masked Language Model Pretraining

Repository for EMNLP 2023 paper [The Distributional Hypothesis Does Not Fully Explain the Benefits of Masked Language Model Pretraining](https://aclanthology.org/2023.emnlp-main.637/).


## Experimenting with Toy Dataset


### Generate synthetic toy data

```
python random_walk.py
```
This will generate both pretraining data and fine-tuning data.


### Pretrain MLM with toy data

To train a model using data *with* the distributional property, run
```
python -m torch.distributed.launch --nproc_per_node 2 ./train_mlm_toy.py --do_train --input_id_setting mix
```

To train a model using data *without* the distributional property, run
```
python -m torch.distributed.launch --nproc_per_node 2 ./train_mlm_toy.py --do_train --input_id_setting shift
```


### Pretrain Word2Vec with toy data

```
python train_wb.py --input_id_setting mix --do_train
python train_wb.py --input_id_setting shift --do_train
```


### Fine-tune models on downstream tasks

```
python ./train_toy.py --train_ratio [train_data_ratio] \
    --load_model [path_to_pretrained_model] \
    --lang1_ratio [lang1_ratio] \
    --max_steps 30000 \
    --per_device_train_batch_size 32 \
    --do_predict \
    --do_train \
    --eval_shift \
    --load_embedding [path_to_embedding] \
```

- `train_data_ratio`: The amount of training data to use. 1.0 means using 100% of the training data. 0.8 means using 80% of the training data.
- `path_to_pretrained_model`: It should be either
    - `../models/random_walk/mlm-mix/` for fine-tuning a model pretrained with data *having* the distributional property.
    - `../models/random_walk/mlm-shift/` for fine-tuning a model pretrained with data *not having* the distributional property. 
- `lang1_ratio`: the ratio of the training examples whose underlying synsets are from D2.
- `path_to_embedding`: Specify this argument to load pretrained token w2v  embeddings.


See `src/train_toy.sh` for examples.


## Experimenting with BERT

### Fine-tuning BERT

```
python ./train_glue.py --dataset mnli --output_dir ../models/mnli-bert-base-uncased --model_name_or_path bert-base-uncased --save_strategy steps --do_train
python ./train_glue.py --dataset sst2 --output_dir ../models/sst2-bert-base-uncased --model_name_or_path bert-base-uncased --save_strategy steps --do_train
```


### Getting the parsings of SST-2

1. Download the dataset from the GLUE website.
2. Use `cut -d $'\t' -f 1 SST-2/dev.tsv  > dev.txt` and `cut -d $'\t' -f 2 SST-2/test.tsv  > test.txt` to extract sentences from the `dev` and the `test` splits.
3. Run Stanford Core NLP `java edu.stanford.nlp.pipeline.StanfordCoreNLP -file test.txt -tokenize.whitespace -ssplit.eolonly -annotators tokenize,pos,parse -outputFormat json`.
4. Move `test.txt.json` and `dev.txt.json` to `mlm-study/data/parses/sst2/`.


### Paraphrasing with Back Translation

To paraphrase the longest subtree, run
```
python ./paraphrase.py mnli
python ./paraphrase.py sst2
```
This will generate 

- `../data/mnli/paraphrased.ds`.
- `../data/sst2/paraphrased.ds`.


To paraphrase shortest important phrase, run
```
python ./paraphrase.py mnli --target_model ../models/mnli-bert-base-uncased/
python ./paraphrase.py sst2 --target_model ../models/sst2-bert-base-uncased/
```

This will generate 
- `../data/mnli/mnli-bert-base-uncased/paraphrased.ds`
- `../data/mnli/sst2-bert-base-uncased/paraphrased.ds`


### Generate Paraphrase with WordNet


```
python ./paraphrase_synonym.py mnli --target_model ../models/mnli-bert-base-uncased/
python ./paraphrase_synonym.py sst2 --target_model ../models/sst2-bert-base-uncased/
```

This will generate
```
../data/sst2/sst2-bert-base-uncased/synonym
../data/mnli/mnli-bert-base-uncased/synonym
```

### Infer Pretrained Models

To infer the fine-tuned models using the task-level and long-phrase-level templates,
```
python infer_mlm.py mnli --mlm_task --mlm_mean --mlm_phrase --mlm_with
```
This will generate `../data/mnli/mlm-outputs-bert-base-uncased/paraphrased.pt`.

To infer the fine-tuned models on the examples perturbed at the short-phrase level,
```
python infer_mlm.py mnli --target_model ../models/mnli-bert-base-uncased/ --mlm_phrase
```
This will generate `../data/mnli/mlm-outputs-bert-base-uncased/mnli-bert-base-uncased/phrase/paraphrased.pt`.


To infer the fine-tuned models on the examples perturbed at the word level,
```
python infer_mlm.py mnli --target_model ../models/mnli-bert-base-uncased --synonym --mlm_phrase
python infer_mlm.py sst2 --target_model ../models/mnli-bert-base-uncased --synonym --mlm_phrase
```
This will generate 
- `../data/mnli/mlm-outputs-bert-base-uncased/mnli-bert-base-uncased/synonym/paraphrased.pt`
- `../data/sst2/mlm-outputs-bert-base-uncased/sst2-bert-base-uncased/synonym/paraphrased.pt`


### Infer Fine-Tuned Models

To infer fine-tuned model on examples perturbed at the long-phrase level,
```
python ./train_glue.py --path_dataset ../data/mnli/paraphrased.ds --do_predict --model_name_or_path ../models/mnli-bert-base-uncased/ --dataset mnli
python ./train_glue.py --path_dataset ../data/sst2/paraphrased.ds --do_predict --model_name_or_path ../models/sst2-bert-base-uncased/ --dataset sst2
```

This will generate predicts in 

- `../data/mnli/mnli-bert-base-uncased/`
- `../data/sst2/sst2-bert-base-uncased/`


To infer fine-tuned model on examples perturbed at the short-phrase level,
```
python ./train_glue.py --path_dataset ../data/mnli/mnli-bert-base-uncased/paraphrased.ds --do_predict --model_name_or_path ../models/mnli-bert-base-uncased/ --dataset mnli
python ./train_glue.py --path_dataset ../data/mnli/sst2-bert-base-uncased/paraphrased.ds --do_predict --model_name_or_path ../models/sst2-bert-base-uncased/ --dataset sst2
```

This will generate predicts in 
- `../data/mnli/mnli-bert-base-uncased/mnli-bert-base-uncased/`
- `../data/sst2/sst2-bert-base-uncased/sst2-bert-base-uncased/`


To infer fine-tuned model on examples perturbed at the word level,
```
python ./train_glue.py --path_dataset ../data/mnli/mnli-bert-base-uncased/synonym/paraphrased.ds --do_predict --model_name_or_path ../models/mnli-bert-base-uncased/ --dataset mnli
python ./train_glue.py --path_dataset ../data/sst2/sst2-bert-base-uncased/synonym/paraphrased.ds --do_predict --model_name_or_path ../models/sst2-bert-base-uncased/ --dataset sst2
```
This will generate predicts in 
- `../data/sst2/sst2-bert-base-uncased/synonym/sst2-bert-base-uncased/`


Run `bash ./infer_fine_tuned.sh` to get the predictions for all the checkpoints.


## Plot the Results

Please check jupyter notebooks in `notebooks/`.

