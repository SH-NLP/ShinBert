# ShinBert

## 1. Build Tokenizer

### Train tokenizer

```bash
$ python3 build_tokenizer.py train
```

### Test tokenizer

```bash
$ python3 build_tokenizer.py test
```

## 2. Pretraining BERT

### Training
```bash
$ cd pretrain
$ python3 pretraining_bert.py
```

## 3. Finetuning
```bash
$ cd finetune
$ python3 run_seq_cls.py --task {$TASK_NAME} --config_file {$CONFIG_FILE}
```
See [finetune/README.md](finetune/README.md) detail
