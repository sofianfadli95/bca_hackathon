# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:23:17 2018

@author: sofyan.fadli
"""

# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from csv file using pandas
data = pd.read_csv('D:\\Lomba Hackaton BCA\\fraud_detection\\data_input\\fraud_train.csv',index_col=False)

data1 = data[['nilai_transaksi','rata_rata_nilai_transaksi', 'maksimum_nilai_transaksi','minimum_nilai_transaksi','rata_rata_jumlah_transaksi']]

# Correlation matrix
corr_mat = data1.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corr_mat, vmax = 1.0, square = True)
plt.show()