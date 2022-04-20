# Sentiment Analysis

> Author: Yangyi Zou

## 1. Dataset Chose

### 1.1 Introduce to Yelp

**Yelp** is a famous merchant review website in the United States. Founded in 2004, it includes merchants in restaurants, shopping centers, hotels, tourism and other fields. Users can score merchants, submit comments and exchange shopping experience on yelp website. If you search yelp for a restaurant or hotel, you can see its brief introduction and the comments of netizens. How many stars will the reviewers give? Usually, the reviewers are consumers who have personally experienced the service of the merchant, and most of the comments are vivid and detailed.

### 1.2 Yelp Reviews

**Yelp reviews** is an open source dataset released by **Yelp** for learning purposes. It contains comments from millions of users, business attributes and more than 200000 photos from multiple metropolitan areas. This is a commonly used global NLP challenge data set, containing 5.2 million comments and 174000 business attributes. The data set download address is:

> https://www.yelp.com/dataset/download

The data in *Yelp reviews* dataset is stored in JSON and SQL formats. Taking JSON format as an example, each review contains the following contents:

```json
{
  // string, 22 character unique review id
  "review_id": "KU_O5udG6zpxOg-VcAEodg",
  
  // string, 22 character unique user id, maps to the user in user.json
  "user_id": "mh_-eMZ6K5RLWhZyISBhwA",
  
  // string, 22 character business id, maps to business in business.json
  "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw",
  
  // integer, star rating
  "stars": 3.0,
  
  // integer, number of useful votes received
  "useful": 0,
  
  // integer, number of funny votes received
  "funny": 0,
  
  // integer, number of cool votes received
  "cool": 0,
  
  // string, the review itself
  "text": "If you decide to eat here, just be aware it is going to take about 2 hours from beginning to end. We have tried it multiple times, because I want to like it! I have been to it\u0027s other locations in NJ and never had a bad experience. \n\nThe food is good, but it takes a very long time to come out. The waitstaff is very young, but usually pleasant. We have just had too many experiences where we spent way too long waiting. We usually opt for another diner or restaurant on the weekends, in order to be done quicker.",
  
  // string, date formatted YYYY-MM-DD
  "date": "2018-07-07 22:09:11"
}
```

Because the *yelp reviews* dataset is relatively large and there is no direct link to download, this paper first downloads the JSON format dataset from [Yelp Dataset | Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) and divides it into ten small files, each containing 100000 comments. The sorted dataset is uploaded to GitHub, which can be obtained from here => [zoeyyyzou/yelp (github.com)](https://github.com/zoeyyyzou/yelp)

## 2. Data cleaning

This project will select 40w reviews from *Yelp reviews* dataset for sentiment analysis, and the number of positive and negative reviews in the selected 40w reviews should be equal. This paper will complete the data cleaning through the following steps：

### 2.1 Load yelp dataset

First, load yelp datasets from JSON format file (which download from  [zoeyyyzou/yelp (github.com) ](https://github.com/zoeyyyzou/yelp)), then use `pandas` to save it to csv format. 

```python
def load_yelp_orig_data():
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
load_yelp_orig_data()
```

Once the above function has been run, you are ready to load it in pandas dataframe for the next steps. 

### 2.2 Exploring data

Fist, display stars counts, then plotting the star distribution to svg.

![star.svg](doc/stars.svg)

Then, mapping from stars to sentiment is done and distribution for each sentiment is plotted. 

> The project stipulates that **stars > 3 is positive** sentiment and **stars < = 3** is negative sentiment

```python
def map_sentiment(stars_received):
    if stars_received <= 3:
        return 0
    else:
        return 1
data_df['sentiment'] = [map_sentiment(x) for x in data_df['stars']]
```

![sentiment.svg](doc/sentiment.svg)

As can be seen from the above figure, a total of **473275** reviews in the current data set are **positive** and **226725** reviews are **negative**.

In order to ensure the same number of positive and negative samples, we use the following code to take 200000 samples from each of the positive and negative samples for subsequent processing, and save to csv file.

```python
def get_top_data(data_df, top_n=200000):
    top_data_df_positive = data_df[data_df['sentiment'] == 1].head(top_n)
    top_data_df_negative = data_df[data_df['sentiment'] == 0].head(top_n)
    top_data_df_small = pd.concat([top_data_df_positive, top_data_df_negative])
    return top_data_df_small
  
data_df = data_df.loc[data_df['text'].str.len() > 20]
data_df = get_top_data(data_df, top_n=200000)
data_df.to_csv(ds_yelp_csv_after_extraction)
```

### 2.3 Data cleaning

Before model training, we need to clean the datasets, we will clean the data according to the following steps.

1. **Remove stop words.**

   In the field of sentiment analysis, because stop words are usually related to emotional expression, this paper does not remove stop words from samples. For example,  `remove_stopwords("I did not like the food!!")` => `I like food!!`。

   In the above example, after removing the stop word, the emotion changes from negative to positive.

2. 

3. 