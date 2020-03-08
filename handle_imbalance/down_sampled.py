# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 17:18:41 2018

@author: sofyan.fadli
"""

import pandas as pd
import numpy as np

from sklearn.utils import resample
# Read dataset
df = pd.read_csv('./balance-scale.csv',
                 names=['balance', 'var1', 'var2', 'var3', 'var4'])

# Display example observations
df.head()

df['balance'].value_counts()

# Transform into binary classification
df['balance'] = [1 if b=='B' else 0 for b in df.balance]

df['balance'].value_counts()

# Separate majority and minority classes
df_majority = df[df.balance == 0]
df_minority = df[df.balance == 1]

# Downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=True,   # sample without replacement
                                   n_samples=49,    # to match minority class
                                   random_state=123)  # reproducible 

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Display new class counts
df_downsampled.balance.value_counts()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Separate input features (X) and target variable (y)
y = df_downsampled.balance
X = df_downsampled.drop('balance', axis=1)

# Train model
clf_2 = LogisticRegression().fit(X, y)

# Predict on training set
pred_y_2 = clf_2.predict(X)

# Is our model still predicting just one class?
print( np.unique( pred_y_2 ) )

# How's our accuracy?
print( accuracy_score(y, pred_y_2) )

from sklearn.metrics import roc_auc_score

# Predict class probabilities
prob_y_2 = clf_2.predict_proba(X)

# Keep only the positive class
prob_y_2 = [p[1] for p in prob_y_2]

prob_y_2[:5]

print( roc_auc_score(y, prob_y_2) )