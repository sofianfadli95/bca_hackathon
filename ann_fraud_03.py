# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:34:48 2018

@author: sofyan.fadli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import metrics


os.chdir("D:\Lomba Hackaton BCA")

####################### PRE PROCESSING ############################
# import dataset
data_train = pd.read_csv("new_data_train.csv", index_col=False)
data_test = pd.read_csv("new_data_test.csv", index_col=False)

data_train = data_train.drop('tipe_kartu', axis=1)
data_train = data_train.drop('tipe_transaksi', axis=1)

data_test = data_test.drop('tipe_kartu', axis=1)
data_test = data_test.drop('tipe_transaksi', axis=1)
# Ubah format column terlebih dahulu
# Ubah beberapa columns integer menjadi string object
data_train['grouped_time'] = data_train.grouped_time.astype(str)
data_test['grouped_time'] = data_test.grouped_time.astype(str)

# Create dummy variables for some columns
data_train = pd.get_dummies(data_train, columns=['grouped_time'], drop_first=True)
data_test = pd.get_dummies(data_test, columns=['grouped_time'], drop_first=True)

"""
# Find the difference in columns between the two datasets
feature_difference = set(data_train) - set(data_test)

# create zero-filled matrix where the rows are equal to the number
# of row in `data_test` and columns equal the number of categories missing (i.e. set difference 
# between relevant `data_train` and `data_test` columns
feature_difference_df = pd.DataFrame(data=np.zeros((data_test.shape[0], len(feature_difference))),
                                     columns=list(feature_difference))

# add "missing" features back to `test
data_test = data_test.join(feature_difference_df)
"""
# select x, y
X_train = data_train.iloc[:, :]
X_train = X_train.drop('flag_transaksi_fraud', axis=1)
X_train = X_train.values
y_train = data_train.iloc[:, 2].values

X_test = data_test.iloc[:, :]
X_test = X_test.drop('flag_transaksi_fraud', axis=1)
X_test = X_test.values
y_test = data_test.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

################################ ANN #####################################
from deeplab import NN

classifier = NN(3, 8, 4, 0.2, 1).ann()
classifier.fit(X_train, y_train, batch_size = 50, nb_epoch = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

############################ Evaluation ##################################
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
evaluate = precision_recall_fscore_support(y_test, y_pred)

metrics.roc_auc_score(y_test, y_pred)
