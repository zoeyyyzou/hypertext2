import json
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scan import getTokens
from tqdm import tqdm
from config import *

# download stopwords
# nltk.download('stopwords')
ps = PorterStemmer()

# get English stopwords
english_stopwords = stopwords.words("english")


def split_yelp():
    """
    Split yelp dataset to train/dev/test => 60%/20%/20%
    :return:
    """
    os.system(f"mkdir -p {ds_yelp}")
    os.system(f"mkdir -p {ds_yelp_split}")
    os.system(f"mkdir -p {ds_yelp_split_after_data_cleaning}")
    with open(f"{ds_yelp_split}/train.txt", "w") as train_file:
        with open(f"{ds_yelp_split}/test.txt", "w") as test_file:
            with open(f"{ds_yelp_split}/dev.txt", "w") as dev_file:
                with open(f"{ds_yelp_split_after_data_cleaning}/train.txt", "w") as train_file_data_cleaning:
                    with open(f"{ds_yelp_split_after_data_cleaning}/test.txt", "w") as test_file_data_cleaning:
                        with open(f"{ds_yelp_split_after_data_cleaning}/dev.txt", "w") as dev_file_data_cleaning:
                            count = 0
                            with tqdm(total=len(ds_yelp_files) * 100000) as bar:  # total表示预期的迭代次数
                                for i in range(len(ds_yelp_files)):
                                    with open(f"{ds_yelp}/{ds_yelp_files[i]}") as file:
                                        line = file.readline()
                                        while line:
                                            item = json.loads(line)
                                            item["text"] = data_cleaning_for_one_document(item["text"])
                                            if not item["text"].strip():
                                                line = file.readline()
                                                count += 1
                                                bar.update(1)
                                                continue
                                            if count % 10 < 2:
                                                test_file.write(line)
                                                test_file_data_cleaning.write(f"{json.dumps(item)}\n")
                                            elif count % 10 < 4:
                                                dev_file.write(line)
                                                dev_file_data_cleaning.write(f"{json.dumps(item)}\n")
                                            else:
                                                train_file.write(line)
                                                train_file_data_cleaning.write(f"{json.dumps(item)}\n")
                                            line = file.readline()
                                            count += 1
                                            bar.update(1)


def data_cleaning_for_one_document(content: str) -> str:
    """
    Do data cleaning for one specific document
    :param content:
    :return:
    """
    # 1. Normalization: convert all token values into lower cases so that different mixes of
    #    the cases for the same word can be all matched (e.g., Book, book, and BOOK are
    #    all converted to “book”).
    content = content.lower()

    tokens = getTokens(content)
    # # 2. Further Filtering: remove certain kinds of tokens, including all numbers and
    # # punctuation marks, since they are unlikely to be used in a query for information
    # # retrieval.
    tokens = [token.value for token in tokens if token.type not in ['NUMBER', 'PUNCTUATION', 'DELIMITERS']]
    # tokens = [token.value for token in tokens]

    # 3. Stop Word Removal: remove all stop words in English (such as “the”, “a”, “in”,
    # and “of”) from the input (see sw_removal.py in A2-Package for an example).
    tokens_nosw = [word for word in tokens if word not in english_stopwords]

    # 4. Stemming: convert the remaining words to their stems. For example, “computer”,
    # “computers”, “compute”, “computed”, “computing”, “computation”, and
    # “computational” can all be reduced to the stem “comput” (see stemming.py in
    # A2-Package for an example).
    # return " ".join(tokens_nosw)
    tokens_final = [ps.stem(word) for word in tokens_nosw]
    return " ".join(tokens_final)


def generate_dataset(dir: str, dir_data_cleaning: str, sampleNum: int, get_label_and_content, train_ratio: float = 0.6,
                     dev_ratio: float = 0.2,
                     test_ratio: float = 0.2):
    """
    generate dataset for hypertext (train.txt, test.txt, dev.txt)
    :param get_label_and_content:
    :param dir:
    :param sampleNum:
    :param train_ratio:
    :param dev_ratio:
    :param test_ratio:
    :return:
    """

    def write_dataset(input_path, output_path: str, limit_count: int):
        with open(output_path, "w") as output_file:
            with open(input_path) as input_file:
                line, count = input_file.readline(), 0
                while line and count < limit_count:
                    output_file.write(get_label_and_content(line))
                    line = input_file.readline()
                    count += 1

    os.system(f"mkdir -p {dir}")
    os.system(f"mkdir -p {dir_data_cleaning}")
    train_count, dev_count, test_count = sampleNum * train_ratio, sampleNum * dev_ratio, sampleNum * test_ratio
    write_dataset(f"{ds_yelp_split}/train.txt", f"{dir}/train.txt", int(train_count))
    write_dataset(f"{ds_yelp_split}/dev.txt", f"{dir}/dev.txt", int(dev_count))
    write_dataset(f"{ds_yelp_split}/test.txt", f"{dir}/test.txt", int(test_count))
    write_dataset(f"{ds_yelp_split_after_data_cleaning}/train.txt", f"{dir_data_cleaning}/train.txt", int(train_count))
    write_dataset(f"{ds_yelp_split_after_data_cleaning}/dev.txt", f"{dir_data_cleaning}/dev.txt", int(dev_count))
    write_dataset(f"{ds_yelp_split_after_data_cleaning}/test.txt", f"{dir_data_cleaning}/test.txt", int(test_count))


def hypertext_get_label_and_content(line: str) -> str:
    item = json.loads(line)
    content = item["text"].strip("\n").replace('\t', ' ').replace("\n", '').replace("\r", '')
    label = 1 if int(item["stars"]) >= 3 else 0
    return f"{content}\t{label}\n"


def fasttext_get_label_and_content(line: str) -> str:
    item = json.loads(line)
    content = item["text"].strip("\n").replace('\t', ' ').replace("\n", '').replace("\r", '')
    label = 1 if int(item["stars"]) >= 3 else 0
    return "%s __label__%d\n" % (content, label)


if __name__ == '__main__':
    split_yelp()

    # Generate datasets of different sizes for HyperText
    for k, v in ds_yelp_hypertext_config.items():
        generate_dataset(v["path"], v["path_after_data_cleaning"], v["count"], hypertext_get_label_and_content)

    # Generate datasets of different sizes for FastText
    for k, v in ds_yelp_fasttext_config.items():
        generate_dataset(v["path"], v["path_after_data_cleaning"], v["count"], fasttext_get_label_and_content)

# python main.py --datasetdir ./datasets/yelp_hypertext_1w --outputdir ./output --dropout 0.0 --require_improvement 6000 --num_epochs 2 --batch_size 32 --max_length 1000 --learning_rate 1.1e-2 --embed_dim 200 --bucket 1500000 --wordNgrams 2 --eval_per_batchs 100 --min_freq 1 --lr_decay_rate 0.96
