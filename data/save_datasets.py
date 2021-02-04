from datasets import load_dataset


data_files = {}
train_file = '../naver_news/naver_news_train.txt'
validation_file = '../naver_news/naver_news_eval.txt'
data_files["train"] = train_file
data_files["validation"] = validation_file


datasets = load_dataset('text', data_files=data_files)
datasets.save_to_disk('/data/aicc1/datasets')
