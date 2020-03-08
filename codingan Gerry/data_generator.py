# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 01:22:49 2018

@author: Gerry
"""

# Import the libraries
import pandas as pd
import os

os.chdir("D:\Lomba Hackaton BCA")
# Importing the dataset
data = pd.read_csv("data_02.csv", index_col=False)
col = str(len(data.columns)-1)
data_fraud = data.loc[data['flag_transaksi_fraud'] == 1].sample(frac = 1)
data_nonfraud = data.loc[data['flag_transaksi_fraud'] == 0].sample(frac = 1)

fraud_test, fraud_train = data_fraud.iloc[:100, :], data_fraud.iloc[100:, :]
nonfraud_test, nonfraud_train = data_nonfraud.iloc[:1000, :], data_nonfraud.iloc[1000:, :]

new_data_train = pd.concat([fraud_train, nonfraud_train], axis = 0, ignore_index = True).sample(frac = 1)
new_data_train.to_csv('new_data_train.csv', index = False)

new_data_test = pd.concat([fraud_test, nonfraud_test], axis = 0, ignore_index = True).sample(frac = 1)
new_data_test.to_csv('new_data_test.csv', index = False)
