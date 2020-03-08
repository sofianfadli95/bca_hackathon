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

os.chdir("D:\Lomba Hackaton BCA")

####################### PRE PROCESSING ############################
# import dataset
data = pd.read_csv("data_02.csv", index_col=False)

# Ubah format column terlebih dahulu
# Ubah beberapa columns integer menjadi string object
data['tipe_kartu'] = data.tipe_kartu.astype(str)
data['tipe_transaksi'] = data.tipe_transaksi.astype(str)
data['kepemilikan_kartu'] = data.kepemilikan_kartu.astype(str)
data['grouped_time'] = data.grouped_time.astype(str)

# filling missing data
data.isnull().sum()
data.fillna(0, inplace = True)

#under sampling
# data_false = data.loc[data['flag_transaksi_fraud'] == 0].sample(n = 910, random_state = 0, axis = 0)
data_false = data.loc[data['flag_transaksi_fraud'] == 0].sample(frac = 0.5, random_state = 0, axis = 0)
data_true = data.loc[data['flag_transaksi_fraud'] == 1]
data = pd.concat([data_false, data_true], axis = 0, ignore_index = True)
data = data.sample(frac = 1)

# over sampling
data = data.append([data.loc[data['flag_transaksi_fraud'] == 1]]*6)
#data = data.append([data]*5)
data = data.sample(frac = 1)

# check data
data.groupby(data.columns[6]).size()

# select x, y
X = data.iloc[:, :6].values
y = data.iloc[:, 6].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
# Mengubah kategorikal data menjadi dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0,1,2,4]) 
X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
# splitting datasets to Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

################################ ANN #####################################
# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

############################ Evaluation ##################################
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
evaluate = precision_recall_fscore_support(y_test, y_pred)

from sklearn.metrics import roc_auc_score
roc_auc_score(y, y_pred)