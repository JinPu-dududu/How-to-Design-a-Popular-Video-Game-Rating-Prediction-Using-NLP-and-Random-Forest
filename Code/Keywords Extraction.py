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


## Get keyword list

from sklearn.utils import shuffle
import pandas as pd
import numpy
from nltk import FreqDist
import nltk
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.porter import *

most_common_all = []
word_key_list_all = []
for k in range(10):
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)
    bins = 10
    cut = pd.cut(df.index,bins,labels=False)
    df['cut'] = cut
    df.groupby('cut').mean()['length']
    
    keylist = []
    for i in range(bins):
        text = ' '.join(df[df['cut']==i]['reviewText'])
        text = text.lower()
        j = json.loads((get_emotion(text)))
        for i in range(len(j['keywords'])):
            keylist.append(j['keywords'][i]['text'])
    keywords = ' '.join(set(keylist))
    print(len(set(keylist)))
            
    p_stemmer = PorterStemmer()
    word_list = nltk.word_tokenize(keywords)
    stopwords = list(STOPWORDS) + ['game','good','great','better','best','love','much','awesome','fun','play','lot','alot','only','little','time']
    words = [p_stemmer.stem(word) for word in word_list if p_stemmer.stem(word) not in stopwords and word not in stopwords and word.isalpha()] 
    # words = [word for word in word_list if word not in stopwords and word.isalpha()] 
    most_common = FreqDist(words).most_common()
    most_common_all.extend(most_common)
    
    word_key_list = []
    for i in most_common:
        key = []
        word = i[0]
        freq = i[1]
        for s in set(keylist):
            if word in [p_stemmer.stem(word) for word in nltk.word_tokenize(s)]:
                key.append(s)
        word_key_list.append((word,freq,key))
    word_key_list_all.extend(word_key_list)



most_common_dict = dict()
for i in most_common_all:
    if most_common_dict.get(i[0]):
        most_common_dict[i[0]] += i[1]
    else:
        most_common_dict[i[0]] = i[1]


sorted_most_common = sorted(most_common_dict.items(),key=lambda x:x[1],reverse=True)
sorted_most_common


word_key_list_dict = dict()
for i in word_key_list_all:
    if word_key_list_dict.get(i[0]):
        word_key_list_dict[i[0]] = list(set(word_key_list_dict[i[0]]+i[2]))
    else:
        word_key_list_dict[i[0]] = i[2]

