import argparse
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str, default='/home/ubuntu/BERT/data/naver_news/news_2_kss/naver_news.txt')
# parser.add_argument("--corpus_file", type=str, default='../data/naver_news/news_2_kss/naver_news00.txt')
parser.add_argument("--vocab_size", type=int, default=32_000)
parser.add_argument("--limit_alphabet", type=int, default=6_000)

args = parser.parse_args()

tokenizer = BertWordPieceTokenizer(
    vocab_file=None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,  # Must be False if cased model
    lowercase=True,
    wordpieces_prefix="##"
)

# tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[args.corpus_file],
    limit_alphabet=args.limit_alphabet,
    vocab_size=args.vocab_size
)

print(tokenizer.save_model("bertwordpiece_32000"))
