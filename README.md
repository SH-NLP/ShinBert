# ShinBert

## 1. Build Tokenizer

### a. Prepair Data

- normalized text data

### b. Train tokenizer

```bash
$$ python3 build_tokenizer.py train
```

### c. Test tokenizer

```bash
$$ python3 build_tokenizer.py test
```

### d. Reference

- [나만의 BERT Wordpiece Vocab 만들기](https://monologg.kr/2020/04/27/wordpiece-vocab/)

## 2. Pretraining BERT

### a. Training

```bash
$$ cd pretrain
$$ python3 pretraining_bert.py
```

### b. Reference

- [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)

## 3. Finetuning

### a. Run

```bash
$$ cd finetune
$$ python3 run_seq_cls.py --task {$TASK_NAME} --config_file {$CONFIG_FILE}
```

### b. Reference

- [2주 간의 KoELECTRA 개발기 - 1부 - Monologg Blog](https://monologg.kr/2020/05/02/koelectra-part1/)
- [2주 간의 KoELECTRA 개발기 - 1부 - Monologg Blog](https://monologg.kr/2020/05/02/koelectra-part1/)

See [finetune/README.md](finetune/README.md) detail
