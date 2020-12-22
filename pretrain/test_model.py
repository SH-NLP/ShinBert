from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./Roberta",
    tokenizer="./bert-wordpiece"
)

text_list = ["aaall lle <mask>.",
             "lqwer ttyui <mask> a sdf gghh."
            ]

for text in text_list:
    print(f'{text} => ')
    print(fill_mask(text))
