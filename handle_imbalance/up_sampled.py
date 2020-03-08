# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:20:35 2018

@author: sofyan.fadli
"""

import pandas as pd
import numpy as np

# Read dataset
df = pd.read_csv('balance-scale.csv',
                 names=['balance', 'var1', 'var2', 'var3', 'var4'])

# Display example observations
df.head()

df['balance'].value_counts()

# Transform into binary classification
df['balance'] = [1 if b=='B' else 0 for b in df.balance]

df['balance'].value_counts()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Separate input features (X) and target variable (y)
y = df.balance
X = df.drop('balance', axis=1)

# Train model
clf_0 = LogisticRegression().fit(X,y)

# Predict on training set
pred_y_0 = clf_0.predict(X)

# How's the accuracy?
print( accuracy_score(pred_y_0, y))

# Should we be excited?
print( np.unique( pred_y_0 ))

### 1. Up-sample Minority Class
from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df[df.balance==0]
df_minority = df[df.balance==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,      # sample with replacement
                                 n_samples=576,     # to match majority class
                                 random_state=123)  # reproducibles results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled.balance.value_counts()

# Separate input features (X) and target variable (y)

y = df_upsampled.balance
X = df_upsampled.drop('balance', axis=1)

# Train model
clf_1 = LogisticRegression().fit(X, y)

# Predict on training set
pred_y_1 = clf_1.predict(X)

# Is our model still predicting just one class?
print( np.unique( pred_y_1 ) )

# How's our accuracy?
print( accuracy_score(y, pred_y_1) )

from sklearn.metrics import roc_auc_score
prob_y_0 = clf_0.predict_proba(X)
prob_y_0 = [p[1] for p in prob_y_0]

print( roc_auc_score(y, prob_y_0) )