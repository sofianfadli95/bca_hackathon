# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 23:54:24 2018

@author: Gerry
"""

import pandas as pd
import numpy as np
import os

os.chdir("D:\BCA")

####################### PRE PROCESSING ############################
# import dataset
data = pd.read_csv("data2.csv")

# filling missing data
data.isnull().sum()
data.fillna(0, inplace = True)

#under sampling
data_false = data.loc[data['4'] == 0].sample(n = 910, random_state = 0, axis = 0)
data_true = data.loc[data['4'] == 1]
data = pd.concat([data_false, data_true], axis = 0, ignore_index = True)
data = data.sample(frac = 1)

# over sampling
data = data.append([data.loc[data['4'] == 1]]*12)
#data = data.append([data]*5)
data = data.sample(frac = 1)

# check data
data.groupby(data.columns[4]).size()

# select x, y
X = data.iloc[:, :4].values
y = data.iloc[:, 4].values

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

classifier = NN(3, 256, 46, 0.2, 1).ann()
classifier.fit(X_train, y_train, batch_size = 50, nb_epoch = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# ANN evaluation
evaluation = NN(3, 64, 233, 0.1, 1).ann_cv(50, 100, X_train, y_train, 10)
mean = evaluation.mean()
variance = evaluation.std()

################################ CNN #####################################
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1],1))
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

classifier = NN(3, 128, 42, 0.1, 1).cnn(32,3,2)
classifier.fit(X_train, y_train, batch_size = 100, nb_epoch = 100)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


############################ Evaluation ##################################
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
evaluate = precision_recall_fscore_support(y_test, y_pred)
