import argparse
from transformers import BertTokenizerFast
from processor.seq_cls import NsmcProcessor


tokenizer_path = '../bertwordpiece_32000'
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
input_texts = ['아 더빙.. 진짜 짜증나네요 목소리', '흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나']
tok = tokenizer(input_texts)
print(tok)

tok = tokenizer.batch_encode_plus(input_texts,
                                  max_length=512,
                                  padding="max_length",
                                  add_special_tokens=True,
                                  truncation=True,
                                  )
print([print(k, v[0][:20]) for k, v in tok.items()])


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data')
parser.add_argument('--task', default='nsmc')
parser.add_argument('--train_file', default='ratings_train.txt')
args = parser.parse_args()
print(args)
print(args.data_dir)
processor = NsmcProcessor(args)
examples = processor.get_examples("train")

input_texts = [examples[0].text_a, examples[1].text_a]
batch_encoding = tokenizer.batch_encode_plus(input_texts,
                                  max_length=512,
                                  padding="max_length",
                                  add_special_tokens=True,
                                  truncation=True,
                                  )
for i, input_text in enumerate(input_texts):
    print(input_text)
    for k in batch_encoding:
        print(k, batch_encoding[k][i][:20])
