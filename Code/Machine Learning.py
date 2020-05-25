#!/usr/bin/env python
# coding: utf-8


import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
with open('review_game.p', 'rb') as f:
    review_game = pickle.load(f)
with open('meta_game.p', 'rb') as f:
    meta_game = pickle.load(f)
with open('product_sentiment_new.p', 'rb') as f:
    product_sentiment = pickle.load(f)
review_game.shape,meta_game.shape,product_sentiment.shape



# review>=5
review_size = review_game.groupby('asin').size()
has_many_review_ids = review_size[review_size>=5].index

product_sentiment_clean = product_sentiment[product_sentiment['asin'].isin(has_many_review_ids)].copy()
product_sentiment_clean.index = range(len(product_sentiment_clean))
product_sentiment_clean.drop('sentimentJson2',axis=1,inplace=True)
has_many_review_ids.shape,product_sentiment_clean.shape



# ratings
review_game_clean = review_game[review_game['asin'].isin(has_many_review_ids)]
ratings = review_game_clean.groupby('asin').mean()['overall']
ratings.shape


# price
meta_game_clean = meta_game[['asin']].copy()
meta_game_clean['true_price'] = meta_game['price'].apply(lambda x:''.join(x[1:].split(','))).astype('float')
meta_game_clean = meta_game_clean[meta_game_clean['asin'].isin(has_many_review_ids)]
meta_game_clean = meta_game_clean.groupby('asin').mean()['true_price'].reset_index()
# decide whether to normalize
meta_game_clean['true_price'] = (meta_game_clean['true_price'] - np.mean(meta_game_clean['true_price']))/np.std(meta_game_clean['true_price'])



# merge all into one dataset
data_ = product_sentiment_clean.merge(meta_game_clean,on='asin')
data = data_.merge(pd.DataFrame(ratings).reset_index(),on='asin')

# fig,ax = plt.subplots(1,1,figsize=(12,6))
# font = {'size': 10}
# matplotlib.rc('font', **font)
sns.distplot(data['overall'])
np.percentile(data['overall'],75),sum(data['overall']<4.5)/sum(data['overall']>=4.4),data.shape



# cut reviews into 2 bins
data_ml = data.copy().set_index('asin')
data_ml['rating'] = np.where(data_ml['overall']>=4.4,1,0)
data_ml.drop('overall',axis=1,inplace=True)
data_ml.head()



# add review size as variable
ids= data_ml.index
review_size = review_size[ids]
review_size = (review_size - np.mean(review_size))/np.std(review_size)
data_ml = data_ml.join(pd.DataFrame(review_size[ids],columns=['n_reviews']))
data = data.join(pd.DataFrame(review_size[ids],columns=['n_reviews']))



data_ml.columns




## Machine Learning

### Feature engineering

data_ml.head()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold 
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_regression,f_classif,SelectKBest

import statsmodels.discrete.discrete_model as sm
import warnings
warnings.filterwarnings("ignore")
import pickle


seed=123
np.random.seed(seed)


def model_cross_validation(model,X,y,folds):
    kfolds=len(folds)
    train_accuracy=np.empty(kfolds)
    validation_accuracy=np.empty(kfolds)
    auc=np.empty(kfolds)
    for idx in range(kfolds):
        train,validation=folds[idx]
        X_train2=X[train]
        y_train2=y[train]
        model.fit(X_train2,y_train2)
        X_validation=X[validation]
        y_validation=y[validation]
        y_score = model.predict_proba(X_validation)[:,1]
        train_accuracy[idx]=np.mean(model.predict(X_train2)==y_train2)
        validation_accuracy[idx]=np.mean(model.predict(X_validation)==y_validation)
        auc[idx] = roc_auc_score(y_validation, y_score)
    return train_accuracy,validation_accuracy,auc


# transform into binary
def binary_substitude(row,sub):
    if isinstance(sub,str):
        if pd.isna(row[sub]):
            return 0
        else:
            return 1
    elif isinstance(sub,tuple):
        judge = 0
        for i in sub:
            if not pd.isna(row[i]):
                judge = 1
        return judge

# use mean value
def keyword_substitude(row,sub):
    return np.nanmean(row[list(sub)])


# remove some factors
data_ml.drop(['player'],axis=1,inplace=True)


substitude = [('old','older'),('real','realistic'),('character','characters'),('control','controls'),('enemy','enemies'),
              ('weapon','weapons'),('review','reviews'),('detail','details'),('attack','attacks'),('animation','animations'),
              ('movie','movies'),('twist','twisted'),('moves','movement','movements'),('magic','magical'),('combo','combos'),
              ('story','storyline','plot')]
for sub in substitude:
    if isinstance(sub,str):
        column_name = 'agg_'+sub
    elif isinstance(sub,tuple):
        column_name = 'agg_'+sub[0]
    data_ml[column_name] = data_ml.apply(lambda row: keyword_substitude(row,sub),axis=1)
    if isinstance(sub,str):
        data_ml.drop(sub,axis=1,inplace=True)
    elif isinstance(sub,tuple):
        for i in sub:
            data_ml.drop(i,axis=1,inplace=True)


substitude2 = ['wii','final fantasy','resident evil','sega gt','star wars','mario','madden','digimon','call of duty',
               ('nintendo','gba'),'god of war','zelda','mech','sonic','golden sun','pokemon','xbox']
for sub in substitude2:
    if isinstance(sub,str):
        column_name = 'series_'+sub
    elif isinstance(sub,tuple):
        column_name = 'series_'+sub[0]
    data_ml[column_name] = data_ml.apply(lambda row: binary_substitude(row,sub),axis=1)
    if isinstance(sub,str):
        data_ml.drop(sub,axis=1,inplace=True)
    elif isinstance(sub,tuple):
        for i in sub:
            data_ml.drop(i,axis=1,inplace=True)


substitude3 = [('action','battle','fight','combat','fighting','fighter','shoot','shooting'),'adventure','rpg','strategy',
               ('sport','sports','basketball','football','baseball','race',"racing","car","wrestling"),('puzzle','puzzles','card'),
               ('kid','kids',"kid's","kids'"),'multiplayer']
for sub in substitude3:
    if isinstance(sub,str):
        column_name = 'type_'+sub
    elif isinstance(sub,tuple):
        column_name = 'type_'+sub[0]
    data_ml[column_name] = data_ml.apply(lambda row: binary_substitude(row,sub),axis=1)
    if isinstance(sub,str):
        data_ml.drop(sub,axis=1,inplace=True)
    elif isinstance(sub,tuple):
        for i in sub:
            data_ml.drop(i,axis=1,inplace=True)



data_ml.fillna(0,inplace=True)
data_ml.head()



### Train-test set split
train,test = train_test_split(data_ml,test_size=0.25,shuffle=True)
train.shape,test.shape  # Too many features compared to the sample size



### Resample

x_train = train.drop('rating',axis=1,inplace=False).values
y_train = train['rating'].values
x_test = test.drop('rating',axis=1,inplace=False).values
y_test = test['rating'].values
names = train.drop('rating',axis=1,inplace=False).columns
x_train.shape,x_test.shape


# from imblearn.over_sampling import SMOTE
# smote = SMOTE(0.5,random_state=2)
# x_train, y_train = smote.fit_sample(x_train, y_train)

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(0.5)
x_train, y_train = rus.fit_sample(x_train, y_train)


x_train.shape,sum(y_train==0),sum(y_train==1)


### K-folds
kf=KFold(5,shuffle=True)     
folds=list(kf.split(x_train))


### Feature Selection Based on Importance

x_train_df = pd.DataFrame(x_train,columns=names)

def simple_feature_importance(X, y, model='reg'):
    score_func = {'reg': f_regression,
                  'clas': f_classif}
    # Score each of the features
    bestfeatures = SelectKBest(score_func=score_func[model], k='all')
    fit = bestfeatures.fit(X, y)
    # Organize and return the scores
    featureScores = pd.DataFrame([X.columns, fit.scores_]).T
    featureScores.columns = ['Feature', 'Score']
    return featureScores.sort_values('Score', ascending=False).set_index('Feature')

### Try to get a threshold based on p-values
feature_names = list(x_train_df.columns)

def f_test(x, y, feature_names, p_cutoff = 0.05):

    from sklearn.feature_selection import f_regression
    num_features = len(feature_names)
    raw_scores, p_values = f_regression(x, y)
    scores = np.zeros(num_features)

    #just keep non statistically significant scores at 0
    for i in range(num_features):
        if (p_values[i] < p_cutoff):
            scores[i] = raw_scores[i]
    return dict(zip(feature_names,scores))

score_data = f_test(x_train_df, y_train, feature_names, p_cutoff = 0.1)
score_data = {key:val for key, val in score_data.items() if val != 0}

importance_scores = simple_feature_importance(x_train_df, y_train, model='clas')
importance_scores = importance_scores[importance_scores['Score']>=min(score_data.values())]
keyword_importance = list(importance_scores.index)

# Get the remaining variables (need to excl. rating to get all IVs.)
keyword_importance[:5]


names2 = keyword_importance
x_train2 = x_train_df[names2].values
x_test2 = test[names2].values
x_train2.shape,x_test2.shape



### Feature Selection - rfe

x_train3 = pd.DataFrame(x_train,columns=names)
x_test3 = pd.DataFrame(x_train,columns=names)
names3 = names

# make a threshold for quasi constant.
threshold = 0.98

# create empty list
quasi_constant_feature = []

# loop over all the columns
for feature in x_train3.columns:

    # calculate the ratio.
    predominant = (x_train3[feature].value_counts() / np.float(len(x_train3))).sort_values(ascending=False).values[0]
    
    # append the column name if it is bigger than the threshold
    if predominant >= threshold:
        quasi_constant_feature.append(feature)   
        
print(quasi_constant_feature)

# drop the quasi constant columns
x_train3.drop(labels=quasi_constant_feature, axis=1,inplace=True)
x_test3.drop(labels=quasi_constant_feature, axis=1,inplace=True)
names3.drop(labels=quasi_constant_feature)

from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=30)

# build the RFE with CV option.
rfe = RFECV(model, min_features_to_select = 3, step = 1 , cv=5, scoring='accuracy')

# fit the RFE to our data.
selection  = rfe.fit(x_train3, y_train)

# print the selected features.
print(x_train3.columns[selection.support_])

names3 = x_train3.columns[selection.support_]
x_train3 = x_train3[names3].values
x_test3 = x_test3[names3].values
x_train3.shape,x_test3.shape



### Random forest

### Random Forest - grid serach
parameters = {
     'n_estimators':range(5,30,5),
     'max_depth': range(1,10,1),
     'min_samples_split': range(10,100,20),
}
model_RF = GridSearchCV(RandomForestClassifier(),parameters,cv=5)
model_RF.fit(x_train,y_train)
best_C_RF = model_RF.best_params_
best_C_RF


### Random Forest - grid serach (feature importance)
model_RF2 = GridSearchCV(RandomForestClassifier(),parameters,cv=5)
model_RF2.fit(x_train2,y_train)
best_C_RF2 = model_RF2.best_params_
best_C_RF2



### Random Forest - grid serach (rfe)
model_RF3 = GridSearchCV(RandomForestClassifier(),parameters,cv=5)
model_RF3.fit(x_train3,y_train)
best_C_RF3 = model_RF3.best_params_
best_C_RF3



### conclude
best_RF = RandomForestClassifier(     
    n_estimators=best_C_RF['n_estimators'],
     max_depth= best_C_RF['max_depth'],
     min_samples_split= best_C_RF['min_samples_split'])
best_RF2 = RandomForestClassifier(     
    n_estimators=best_C_RF2['n_estimators'],
     max_depth= best_C_RF2['max_depth'],
     min_samples_split= best_C_RF2['min_samples_split'])
best_RF3 = RandomForestClassifier(     
    n_estimators=best_C_RF3['n_estimators'],
     max_depth= best_C_RF3['max_depth'],
     min_samples_split= best_C_RF3['min_samples_split'])
models = [best_RF,best_RF2,best_RF3]

columns=['RF','RF-feature importance','RF-rfe']
vacc = pd.DataFrame(columns=columns)
auc = pd.DataFrame(columns=columns)
train_accuracy,validation_accuracy,auc_ = model_cross_validation(models[0],x_train,y_train,folds)
vacc[columns[0]]= validation_accuracy
auc[columns[0]] = auc_
train_accuracy2,validation_accuracy2,auc_2 = model_cross_validation(models[1],x_train2,y_train,folds)
vacc[columns[1]]= validation_accuracy2
auc[columns[1]] = auc_2
train_accuracy3,validation_accuracy3,auc_3 = model_cross_validation(models[2],x_train3,y_train,folds)
vacc[columns[2]]= validation_accuracy3
auc[columns[2]] = auc_3


fig,axes = plt.subplots(1,2,figsize=(12,6),sharey=True)
vacc.boxplot(ax=axes[0])
auc.boxplot(ax=axes[1])
axes[0].set_ylabel('Accuracy')
axes[1].set_ylabel('AUC')



### Interpretation

rf = best_RF.fit(x_train,y_train)

from sklearn.inspection import permutation_importance

result = permutation_importance(rf, x_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

fig,ax= plt.subplots(1,1,figsize=(12,18))
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=names[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()

