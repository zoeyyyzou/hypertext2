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

### 2.1 