# dataset save path
import os

dataset_dir = "../datasets"

# yelp dataset
ds_yelp = f"{dataset_dir}{os.sep}yelp"
ds_yelp_split = f"{dataset_dir}{os.sep}yelp_split"
ds_yelp_split_after_data_cleaning = f"{dataset_dir}{os.sep}yelp_split_data_cleaning"
ds_yelp_files = ["yelp_review_0.json",
                 "yelp_review_1.json",
                 "yelp_review_2.json",
                 "yelp_review_3.json",
                 "yelp_review_4.json",
                 "yelp_review_5.json",
                 "yelp_review_6.json",
                 # "yelp_review_7.json",
                 # "yelp_review_8.json",
                 # "yelp_review_9.json"
                 ]
ds_yelp_csv = f"{dataset_dir}{os.sep}yelp_csv.csv"
ds_yelp_csv_after_extraction = f"{dataset_dir}{os.sep}yelp_csv_after_extraction.csv"
ds_yelp_csv_after_data_cleaning = f"{dataset_dir}{os.sep}ds_yelp_csv_after_data_cleaning.csv"

# yelp for hypertext_model
ds_yelp_hypertext_config = {
    # "yelp_hypertext_1000": {
    #     "path": f"{dataset_dir}{os.sep}yelp_hypertext_1000",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_hypertext_1000_data_cleaning",
    #     "count": 1000
    # },
    "yelp_hypertext_1w": {
        "dirname": "yelp_hypertext_1w",
        "path": f"{dataset_dir}{os.sep}yelp_hypertext_1w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_hypertext_1w_data_cleaning",
        "count": 10000
    },
    "yelp_hypertext_10w": {
        "dirname": "yelp_hypertext_10w",
        "path": f"{dataset_dir}{os.sep}yelp_hypertext_10w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_hypertext_10w_data_cleaning",
        "count": 100000
    },
    "yelp_hypertext_20w": {
        "dirname": "yelp_hypertext_20w",
        "path": f"{dataset_dir}{os.sep}yelp_hypertext_20w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_hypertext_20w_data_cleaning",
        "count": 200000
    },
    "yelp_hypertext_40w": {
        "dirname": "yelp_hypertext_40w",
        "path": f"{dataset_dir}{os.sep}yelp_hypertext_40w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_hypertext_40w_data_cleaning",
        "count": 400000
    },
    # "yelp_hypertext_60w": {
    #     "path": f"{dataset_dir}{os.sep}yelp_hypertext_60w",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_hypertext_60w_data_cleaning",
    #     "count": 600000
    # },
    # "yelp_hypertext_80w": {
    #     "path": f"{dataset_dir}{os.sep}yelp_hypertext_80w",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_hypertext_80w_data_cleaning",
    #     "count": 800000
    # },
    # "yelp_hypertext_100w": {
    #     "path": f"{dataset_dir}{os.sep}yelp_hypertext_100w",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_hypertext_100w_data_cleaning",
    #     "count": 1000000
    # },
}

# yelp for fasttext_model
ds_yelp_fasttext_config = {
    # "yelp_fasttext_1000": {
    #     "path": f"{dataset_dir}{os.sep}yelp_fasttext_1000",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_fasttext_1000_data_cleaning",
    #     "count": 1000
    # },
    "yelp_fasttext_1w": {
        "dirname": "yelp_fasttext_1w",
        "path": f"{dataset_dir}{os.sep}yelp_fasttext_1w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_fasttext_1w_data_cleaning",
        "count": 10000
    },
    "yelp_fasttext_10w": {
        "dirname": "yelp_fasttext_10w",
        "path": f"{dataset_dir}{os.sep}yelp_fasttext_10w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_fasttext_10w_data_cleaning",
        "count": 100000
    },
    "yelp_fasttext_20w": {
        "dirname": "yelp_fasttext_20w",
        "path": f"{dataset_dir}{os.sep}yelp_fasttext_20w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_fasttext_20w_data_cleaning",
        "count": 200000
    },
    "yelp_fasttext_40w": {
        "dirname": "yelp_fasttext_40w",
        "path": f"{dataset_dir}{os.sep}yelp_fasttext_40w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_fasttext_40w_data_cleaning",
        "count": 400000
    },
    # "yelp_fasttext_60w": {
    #     "path": f"{dataset_dir}{os.sep}yelp_fasttext_60w",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_fasttext_60w_data_cleaning",
    #     "count": 600000
    # },
    # "yelp_fasttext_80w": {
    #     "path": f"{dataset_dir}{os.sep}yelp_fasttext_80w",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_fasttext_80w_data_cleaning",
    #     "count": 800000
    # },
    # "yelp_fasttext_100w": {
    #     "path": f"{dataset_dir}{os.sep}yelp_fasttext_100w",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_fasttext_100w_data_cleaning",
    #     "count": 1000000
    # },
}

# yelp for word2vec_model
ds_yelp_word2vec_config = {
    # "yelp_word2vec_1000": {
    #     "path": f"{dataset_dir}{os.sep}yelp_word2vec_1000",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_word2vec_1000_data_cleaning",
    #     "count": 1000
    # },
    "yelp_word2vec_1w": {
        "dirname": "yelp_word2vec_1w",
        "path": f"{dataset_dir}{os.sep}yelp_word2vec_1w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_word2vec_1w_data_cleaning",
        "count": 10000
    },
    "yelp_word2vec_10w": {
        "dirname": "yelp_word2vec_10w",
        "path": f"{dataset_dir}{os.sep}yelp_word2vec_10w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_word2vec_10w_data_cleaning",
        "count": 100000
    },
    "yelp_word2vec_20w": {
        "dirname": "yelp_word2vec_20w",
        "path": f"{dataset_dir}{os.sep}yelp_word2vec_20w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_word2vec_20w_data_cleaning",
        "count": 200000
    },
    "yelp_word2vec_40w": {
        "dirname": "yelp_word2vec_40w",
        "path": f"{dataset_dir}{os.sep}yelp_word2vec_40w",
        "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_word2vec_40w_data_cleaning",
        "count": 400000
    },
    # "yelp_word2vec_60w": {
    #     "path": f"{dataset_dir}{os.sep}yelp_word2vec_60w",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_word2vec_60w_data_cleaning",
    #     "count": 600000
    # },
    # "yelp_word2vec_80w": {
    #     "path": f"{dataset_dir}{os.sep}yelp_word2vec_80w",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_word2vec_80w_data_cleaning",
    #     "count": 800000
    # },
    # "yelp_word2vec_100w": {
    #     "path": f"{dataset_dir}{os.sep}yelp_word2vec_100w",
    #     "path_after_data_cleaning": f"{dataset_dir}{os.sep}yelp_word2vec_100w_data_cleaning",
    #     "count": 1000000
    # },
}
