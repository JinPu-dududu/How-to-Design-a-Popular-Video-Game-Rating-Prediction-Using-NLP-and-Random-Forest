#!/usr/bin/env python
# coding: utf-8

import pickle
# Restore from a file
f = open('review_game.p', 'rb')
df = pickle.load(f)

df['reviewText'] = df['reviewText'].astype(str)
df['reviewText'].head()


apikey = '...'
url = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/42f361bc-4cce-4639-b79f-fede5ffdbf02'


#https://cloud.ibm.com/apidocs/natural-language-understanding/natural-language-understanding
# !pip install --upgrade "ibm-watson>=4.3.0"
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions, EmotionOptions, SentimentOptions

authenticator = IAMAuthenticator(apikey)
natural_language_understanding = NaturalLanguageUnderstandingV1(version='2019-07-12',authenticator=authenticator)

natural_language_understanding.set_service_url(url)


## Sentiment Analysis

with open('keyword2.txt','r') as f:
    keylist3 = f.read()
f.close()
keylist3 = keylist3.split(',')
len(keylist3)

ids = df['asin'].unique()

product_review = []
for i in ids:
    review = ' '.join(df[df['asin']==i]['reviewText'])
    product_review.append((i,review))


product_review = pd.DataFrame(product_review,columns=['asin','review'])
product_review.head()


product_review['length'] = product_review['review'].apply(lambda x:len(x))


fig,ax = plt.subplots(1,1,figsize=(18,6))
product_review['length'].plot()
plt.ylim(0,100000)


def get_sentiment_keywords(text_,keylist):
    if len(text_) < 15:
        return ''
    else:
        try:
            response = natural_language_understanding.analyze(language='en',text=text_.lower(),features=Features(sentiment=SentimentOptions(targets=keylist))).get_result()
            string = json.dumps(response)
            return string
        except:
            print(text_)
            return('')


product_review['sentimentJson2'] = product_review.apply(lambda row:get_sentiment_keywords(row.review,keylist3), axis=1)


product_sentiment = product_review[product_review['sentimentJson2']!=""][['asin','sentimentJson2']]

def extract_sentiment(x,key):
    targets = json.loads(x)['sentiment']['targets']
    for target in targets:
        if target['text'] == key:
            return target['score']
    else:
        return 0
        
for key in keylist3:
    product_sentiment[key] =  product_sentiment['sentimentJson2'].apply(lambda x: extract_sentiment(x,key))

product_sentiment.columns

