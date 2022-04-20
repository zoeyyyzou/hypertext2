from config import *
import pandas as pd
import matplotlib.pyplot as plt
from scan import getTokens
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.utils import shuffle
import numpy as np

# download stopwords
# nltk.download('stopwords')
ps = PorterStemmer()

# get English stopwords
english_stopwords = stopwords.words("english")


def map_sentiment(stars_received):
    """
    Function to map stars to sentiment
    Args:
        stars_received:

    Returns:
    """
    if stars_received <= 3:
        return 0
    else:
        return 1


def get_top_data(data_df, top_n=200000):
    """
    Function to retrieve top few number of each category
    Args:
        data_df:
        top_n:

    Returns:

    """
    top_data_df_positive = data_df[data_df['sentiment'] == 1].head(top_n)
    top_data_df_negative = data_df[data_df['sentiment'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative])
    return top_data_df_small


def load_yelp_orig_data():
    """
    Load json format yelp dataset, save to csv format
    Returns:

    """
    data = []
    for file in ds_yelp_files:
        with open(f"{ds_yelp}{os.sep}{file}", "r") as f:
            data += f.readlines()

    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)

    data_json_str = "[" + ','.join(data) + "]"

    # now, load it into pandas
    data_df = pd.read_json(data_json_str)
    data_df.to_csv(ds_yelp_csv)


def data_display_and_extraction():
    # 1. load yelp dataset
    data_df = pd.read_csv(ds_yelp_csv)

    # 2. display star counts, then plotting the star distribution to svg
    print("\nStar distribution:")
    print(data_df['stars'].value_counts())
    pd.value_counts(data_df['stars']).plot.bar(title="Star distribution in yelp")
    plt.xlabel("Star")
    plt.ylabel("No. of rows in yelp")
    plt.savefig("stars.svg", bbox_inches='tight', pad_inches=0)  # 存储路径
    plt.close()

    # 3. map star to sentiment
    data_df['sentiment'] = [map_sentiment(x) for x in data_df['stars']]

    # 4. display sentiment counts, then plotting the sentiment distribution to svg
    print("\nSentiment distribution:")
    print(data_df['sentiment'].value_counts())
    pd.value_counts(data_df['sentiment']).plot.bar(title="Sentiment distribution in yelp")
    plt.xlabel("Sentiment")
    plt.ylabel("No. of rows in yelp")
    plt.savefig("sentiment.svg", bbox_inches='tight', pad_inches=0)  # 存储路径
    plt.close()

    # 5. Function call to get the top 200000 from each sentiment
    top_data_df_small = get_top_data(data_df, top_n=200000)

    print("\nAfter segregating and taking equal number of rows for each sentiment:")
    print(top_data_df_small['sentiment'].value_counts())

    # 6. remove all other columns, except 'text' and 'sentiment'
    top_data_df_small[["text", "sentiment"]].to_csv(ds_yelp_csv_after_extraction)


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


def data_cleaning():
    # 1. load yelp dataset
    data_df = pd.read_csv(ds_yelp_csv_after_extraction)

    # 2. do data cleaning for each sample
    with tqdm(total=data_df.shape[0]) as bar:
        for index, row in data_df.iterrows():
            row["text"] = data_cleaning_for_one_document(row["text"])
            bar.update()

    data_df.to_csv(ds_yelp_csv_after_data_cleaning)


def split_dataset(sampleNum: int, train_ratio: float = 0.6, dev_ratio: float = 0.2, test_ratio: float = 0.2):
    """
    Split dataset to train/dev/test
    Args:
        sampleNum:
        train_ratio:
        dev_ratio:
        test_ratio:

    Returns:
    """

    data_df = pd.read_csv(ds_yelp_csv_after_data_cleaning)
    train_count, dev_count, test_count = int(sampleNum * train_ratio), int(sampleNum * dev_ratio), int(
        sampleNum * test_ratio)
    train_count_each, dev_count_each, test_count_each = int(train_count / 2), int(dev_count / 2), int(test_count / 2)
    negative_datas = data_df.query("sentiment==0")
    positive_datas = data_df.query("sentiment==1")
    train_df = shuffle(pd.concat([negative_datas.iloc[:train_count_each], positive_datas.iloc[:train_count_each]]))
    dev_df = shuffle(pd.concat([negative_datas.iloc[train_count_each:train_count_each + dev_count_each],
                                positive_datas.iloc[train_count_each:train_count_each + dev_count_each]]))
    test_df = shuffle(pd.concat(
        [negative_datas.iloc[train_count_each + dev_count_each:train_count_each + dev_count_each + test_count_each],
         positive_datas.iloc[train_count_each + dev_count_each:train_count_each + dev_count_each + test_count_each]]))

    return train_df, dev_df, test_df


def hypertext_get_label_and_content(row) -> str:
    content = row["text"].strip("\n").replace('\t', ' ').replace("\n", '').replace("\r", '')
    label = row['sentiment']
    return f"{content}\t{label}\n"


def fasttext_get_label_and_content(row) -> str:
    content = row["text"].strip("\n").replace('\t', ' ').replace("\n", '').replace("\r", '')
    label = row['sentiment']
    return "%s __label__%d\n" % (content, label)


def generate_hypertext_and_fasttext_dataset(output_dir: str, get_label_and_content, train_df, dev_df, test_df):
    """
    generate dataset for hypertext and fasttext (train.txt, test.txt, dev.txt)
    Args:
        output_dir:
        get_label_and_content:
        train_df:
        dev_df:
        test_df:

    Returns:

    """
    os.system(f"mkdir -p {output_dir}")
    with open(f"{output_dir}{os.sep}train.txt", "w") as f:
        for index, row in train_df.iterrows():
            f.write(get_label_and_content(row))
    with open(f"{output_dir}{os.sep}dev.txt", "w") as f:
        for index, row in dev_df.iterrows():
            f.write(get_label_and_content(row))
    with open(f"{output_dir}{os.sep}test.txt", "w") as f:
        for index, row in test_df.iterrows():
            f.write(get_label_and_content(row))


if __name__ == '__main__':
    # 1. load json format yelp dataset, then convert to csv format
    # load_yelp_orig_data()

    # 2. display dataset detail, then extract an equal number of samples for each sentiment
    # data_display_and_extraction()

    # 3. do data cleaning
    # data_cleaning()

    # 4. generate datasets
    # Generate datasets of different sizes for HyperText
    for k, v in ds_yelp_hypertext_config.items():
        train_df, dev_df, test_df = split_dataset(v["count"])
        generate_hypertext_and_fasttext_dataset(v["path"], hypertext_get_label_and_content,
                                                train_df, dev_df, test_df)

    # Generate datasets of different sizes for FastText
    for k, v in ds_yelp_fasttext_config.items():
        train_df, dev_df, test_df = split_dataset(v["count"])
        generate_hypertext_and_fasttext_dataset(v["path"], fasttext_get_label_and_content,
                                                train_df, dev_df, test_df)

