# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 19:18:03 2018

@author: sofyan.fadli
"""

import pandas as pd
import numpy as np

# Read dataset
df = pd.read_csv('balance-scale.csv',
                 names=['balance', 'var1', 'var2', 'var3', 'var4'])

df['balance'].value_counts()

# Transform into binary classification
df['balance'] = [1 if b=='B' else 0 for b in df.balance]

df['balance'].value_counts()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# Separate input features (X) and target variable (y)
y = df.balance
X = df.drop('balance', axis=1)

# Train model
clf_3 = SVC(kernel='linear',
            class_weight='balanced',  # penalize
            probability=True)

clf_3.fit(X, y)

# Predict on training set
pred_y_3 = clf_3.predict(X)

# Is our model still predicting just one class?
print( np.unique( pred_y_3 ) )

# How's our accuracy?
print( accuracy_score(y, pred_y_3) )

# What about AUROC?
prob_y_3 = clf_3.predict_proba(X)
prob_y_3 = [p[1] for p in prob_y_3]
print( roc_auc_score(y, prob_y_3) )