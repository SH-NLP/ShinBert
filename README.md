# ShinBert

## 1. Build Tokenizer

### a. Prepair Data

- normalized text data

### b. Train tokenizer

```bash

Edit parameters in build_tokenizer.py and run

```bash
$$ cd tokenizer
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

Edit parameters in pretraining_bert.py and run

```bash
$$ cd pretrain
$$ python3 pretraining_bert.py
```

### b. Reference

- [How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)
- [colab](https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb)

## 3. Finetuning

### a. Run

Edit config.json and run

```bash
$$ cd finetune
$$ python3 run_seq_cls.py --task {$TASK_NAME} --config_file {$CONFIG_FILE}
```

### b. Reference

- [2주 간의 KoELECTRA 개발기 - 1부 - Monologg Blog](https://monologg.kr/2020/05/02/koelectra-part1/)
- [2주 간의 KoELECTRA 개발기 - 2부 - Monologg Blog](https://monologg.kr/2020/05/02/koelectra-part2/)

See [finetune/README.md](finetune/README.md) detail
