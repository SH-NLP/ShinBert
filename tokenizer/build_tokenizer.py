""" build tokenizer
Usage:
    build_tokenizer.py (train|test)

"""
from docopt import docopt
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, AutoTokenizer


def main(args):
    print(args)
    if args['train']:
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=True,  # Must be False if cased model
            lowercase=True,
            wordpieces_prefix="##"
        )

        tokenizer.train(
            files=['/data2/BERT/data/naver_news/news_3_preprocessed/naver_news.txt'],
            limit_alphabet=6000,
            vocab_size=32000
        )

        print(tokenizer.save_model("../BertWordPieceTokenizer_32000"))

    elif args['test']:
        test_str = '나는 워드피스 토크나이저를 써요. 성능이 좋은지 테스트 해보려 합니다.'

        print("=========== tokenizer ===========")
        tokenizer = BertWordPieceTokenizer("../BertWordPieceTokenizer_32000/vocab.txt")
        print(tokenizer)
        encoded_str = tokenizer.encode(test_str)
        print('encoding: ', encoded_str.ids)
        decoded_str = tokenizer.decode(encoded_str.ids)
        print(decoded_str)

        print("=========== BertTokenizer ===========")
        tokenizer = BertTokenizer("../BertWordPieceTokenizer_32000/vocab.txt")
        print(tokenizer)
        encoded_str = tokenizer.encode(test_str)
        print('encoding: ', encoded_str)
        decoded_str = tokenizer.decode(encoded_str)
        print(decoded_str)

        print("=========== BertTokenizer2 ===========")
        tokenizer = BertTokenizer.from_pretrained("../BertWordPieceTokenizer_32000")
        print(tokenizer)
        encoded_str = tokenizer.encode(test_str)
        print('encoding: ', encoded_str)
        decoded_str = tokenizer.decode(encoded_str)
        print(decoded_str)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
