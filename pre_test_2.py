# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 09:43:18 2018

@author: sofyan.fadli
"""

import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

data_train = pd.read_csv('./train1.csv')
data_train.columns

data_test = pd.read_csv('./test1.csv')
data_test.columns

data_train.sort_values('revol_bal', inplace=True)

plot({
    'data': [{'x':data_train['revol_bal'], 'y': data_train['annual_inc'], 'name':'Revolving Balance'}] ,
    'layout': {'font': dict(size=14), 'title':'Revolving Balance terhadap Annual Income'}    
},
show_link=False)

data_train.sort_values('annual_inc', inplace=True)

plot({
    'data': [{'x':data_train['annual_inc'], 'y': data_train['revol_bal'], 'name':'Annual Income'}] ,
    'layout': {'font': dict(size=14), 'title':'Annual Income terhadap Revolving Balance'}    
},
show_link=False)

set(data_train['purpose'])
data_train.groupby('purpose').count()

filter1 = (data_train['purpose'] == 'debt_consolidation') & (data_train['not_paid'] == 1)
data_train[filter1].groupby('not_paid').count()

data_train.groupby('purpose').count()['annual_inc']/ data_train.groupby('purpose').count()['annual_inc'].sum() * 100

# Create Regression Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Handling Categorical Data
# Handling Categorical Data: home_ownership
map_home_ownership = dict(zip(set(data_train['home_ownership']), [1,2,3]))
map_home_ownership

data_train['home_ownership'] = data_train['home_ownership'].apply(lambda x: map_home_ownership[x])
data_test['home_ownership'] = data_test['home_ownership'].apply(lambda x: map_home_ownership[x])

# Handling Categorical Data: purpose
map_purpose = dict(zip(set(data_train['purpose']), [1,2,3,4,5]))
map_purpose

data_train['purpose'] = data_train['purpose'].apply(lambda x: map_purpose[x])
data_test['purpose'] = data_test['purpose'].apply(lambda x: map_purpose[x])

# Training the regression model
predictors = ['purpose', 'int_rate', 'installment', 'annual_inc', 'verified', 'home_ownership', 'grdCtoA']
X_train = data_train[predictors]
y_train = data_train['not_paid']
model.fit(X_train, y_train)

# Testing the model
X_test = data_test[predictors]
y_test = data_test['not_paid']
predictions = model.predict(X_test)  # will output array with integer values.

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

for col in X_test.columns:
    cor = data_test['not_paid'].corr(data_test[col])
    if cor < 0:
        print(col, cor)

import numpy as np
model2 = model
coef2 = dict(zip(predictors, model2.coef_[0]))
coef2['grdCtoA'] = -0.3298

model2.coef_ = np.array([list(coef2.values())])
print(model2.coef_)

condition1 = (X_test['grdCtoA'] == 1)
condition2 = (X_test['grdCtoA'] != 1)

X_test1 = X_test[condition1]
X_test2 = X_test[condition2]

X_test1.count()
X_test2.count()
X_test.count()

cek = dict(zip(model2.predict_proba(X_test1)[:,1], model2.predict(X_test1)))
cek

model2.predict_proba(X_test1)[:,1].sum()/len(model2.predict_proba(X_test1)[:,1])

np.exp(model2.coef_)

coef2 = dict(zip(predictors, model2.coef_[0]))
coef2['grdCtoA']

'{:.4f}'.format(np.exp(coef2['grdCtoA']))

recall = 97/(54+97)
precision = 97/(68+97)
accuracy = (97+93)/(93+54+68+97)
