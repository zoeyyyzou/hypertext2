from config import *
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer

porter_stemmer = PorterStemmer()


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
    data_df = data_df.loc[data_df['text'].str.len() > 20]
    top_data_df_small = get_top_data(data_df, top_n=200000)

    print("\nAfter segregating and taking equal number of rows for each sentiment:")
    print(top_data_df_small['sentiment'].value_counts())

    # 6. remove all other columns, except 'text' and 'sentiment'
    top_data_df_small[["text", "sentiment"]].to_csv(ds_yelp_csv_after_extraction)


def data_cleaning():
    # 1. load yelp dataset
    data_df = pd.read_csv(ds_yelp_csv_after_extraction)

    # 2. Tokenization: Tokenization is the process in which the sentence/text is split into array of words called tokens.
    # This helps to do transformations on each words separately and this is also required to transform words to numbers.
    #
    # Gensim’s simple_preprocess allows you to convert text to lower case and remove punctuations. It has min and max
    # length parameters as well which help to filter out rare words and most commonly words which will fall in that
    # range of lengths.
    print("Start data cleaning => Tokenization")
    # data_df = data_df.head(100)
    data_df['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in tqdm(data_df['text'])]
    # print(data_df.query("81876"))
    # return
    # 3. Stemming: Stemming process reduces the words to its’ root word. Unlike Lemmatization which uses grammar rules
    # and dictionary for mapping words to root form, stemming simply removes suffixes/prefixes. Stemming is widely used
    # in the application of SEOs, Web search results, and information retrieval since as long as the root matches in the
    # text somewhere it helps to retrieve all the related documents in the search
    print("Start data cleaning => Stemming")
    data_df['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in
                                 tqdm(data_df['tokenized_text'])]

    print("Start data cleaning => Merge token to text")
    data_df['text'] = [" ".join(tokens) for tokens in tqdm(data_df['tokenized_text'])]

    # 4. save to scv
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
            res = get_label_and_content(row)
            if res:
                f.write(res)
    with open(f"{output_dir}{os.sep}dev.txt", "w") as f:
        for index, row in dev_df.iterrows():
            res = get_label_and_content(row)
            if res:
                f.write(res)
    with open(f"{output_dir}{os.sep}test.txt", "w") as f:
        for index, row in test_df.iterrows():
            res = get_label_and_content(row)
            if res:
                f.write(res)


def generate_word2vec_dataset(output_dir: str, train_df, dev_df, test_df):
    """
    Generate dataset for word2vec
    Args:
        output_dir:
        train_df:
        dev_df:
        test_df:

    Returns:

    """
    os.system(f"mkdir -p {output_dir}")
    train_df.to_csv(f"{output_dir}{os.sep}train.csv")
    dev_df.to_csv(f"{output_dir}{os.sep}dev.csv")
    test_df.to_csv(f"{output_dir}{os.sep}test.csv")


if __name__ == '__main__':
    # 1. load json format yelp dataset, then convert to csv format
    print("1. Start load yelp dataset")
    load_yelp_orig_data()

    # 2. display dataset detail, then extract an equal number of samples for each sentiment
    print("2. Start display and extraction")
    data_display_and_extraction()

    # 3. do data cleaning
    print("3. Start data cleaning")
    # data_cleaning()

    # 4. generate datasets
    print("4. Start generate datasets for model")
    # Generate datasets of different sizes for HyperText
    print("4.1 Generate datasets of different sizes for HyperText")
    for k, v in ds_yelp_hypertext_config.items():
        train_df, dev_df, test_df = split_dataset(v["count"])
        generate_hypertext_and_fasttext_dataset(v["path"], hypertext_get_label_and_content,
                                                train_df, dev_df, test_df)

    # Generate datasets of different sizes for FastText
    print("4.2 Generate datasets of different sizes for FastText")
    for k, v in ds_yelp_fasttext_config.items():
        train_df, dev_df, test_df = split_dataset(v["count"])
        generate_hypertext_and_fasttext_dataset(v["path"], fasttext_get_label_and_content,
                                                train_df, dev_df, test_df)

    # Generate datasets of different sizes for word2vec
    print("4.3 Generate datasets of different sizes for word2vec")
    for k, v in ds_yelp_word2vec_config.items():
        train_df, dev_df, test_df = split_dataset(v["count"])
        generate_word2vec_dataset(v["path"], train_df, dev_df, test_df)
