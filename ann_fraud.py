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
data = pd.read_csv("data_02.csv", index_col=False)
# Ubah format column terlebih dahulu
# Ubah beberapa columns integer menjadi string object
data['tipe_kartu'] = data.tipe_kartu.astype(str)
data['tipe_transaksi'] = data.tipe_transaksi.astype(str)
data['grouped_time'] = data.grouped_time.astype(str)

# filling missing data
data.fillna(0, inplace = True)

#under sampling
data_false = data.loc[data['flag_transaksi_fraud'] == 0].sample(n = 910, random_state = 0, axis = 0)
data_true = data.loc[data['flag_transaksi_fraud'] == 1]
data = pd.concat([data_false, data_true], axis = 0, ignore_index = True)
data = data.sample(frac = 1)
data.isnull().sum()

# over sampling
data = data.append([data.loc[data['flag_transaksi_fraud'] == 1]]*12)
#data = data.append([data]*5)
data = data.sample(frac = 1)

# check data
data.groupby(data.columns[5]).size()

# select x, y
X = data.iloc[:, :5].values
y = data.iloc[:, 5].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

# Mengubah kategorikal data menjadi dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0,1,2]) 
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
from deeplab import NN

classifier = NN(5, 64, 41, 0.2, 1).ann()
classifier.fit(X_train, y_train, batch_size = 50, nb_epoch = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

############################ Evaluation ##################################
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
evaluate = precision_recall_fscore_support(y_test, y_pred)

metrics.roc_auc_score(y_test, y_pred)
