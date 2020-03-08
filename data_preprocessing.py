# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 23:33:31 2018

@author: sofyan.fadli
"""

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import pandas as pd
import seaborn as sns

# Importing the dataset
data = pd.read_csv('D:\\Lomba Hackaton BCA\\fraud_detection\\data_input_csv\\data_input\\fraud_train.csv', index_col=False)

# Mencari nilai max dari transaksi
data['nilai_transaksi'].max()
# Mencari nilai rata-rata dari transaksi
data['rata_rata_nilai_transaksi'].max()

fraud_transaction = data[data['flag_transaksi_fraud'] == 1]
non_fraud = data[data['flag_transaksi_fraud'] == 0]
file_name = "real_fraud.csv"
fraud_transaction.to_csv(file_name, sep=',', encoding='utf-8', index=False)

fraud_transaction.groupby('tipe_kartu').count()
fraud_transaction.groupby('tipe_transaksi').count()

# Hanya bisa 1 graph saja yg di plot
size, scale = 1000, 50
tipe_transaksi_non = non_fraud['tipe_transaksi']
tipe_transaksi_fraud = fraud_transaction['tipe_transaksi']
tipe_transaksi_fraud.plot.hist(grid=True, bins=10, rwidth=0.9,
                   color='#607c8e')
                   
plt.title('Amounts of Tipe Transaksi')
plt.xlabel('Tipe Transaksi')
plt.ylabel('Counts')
plt.grid(axis='y', alpha=0.75)

# Bisa utk plot 2 graph
tipe_transaksi_non = non_fraud['tipe_transaksi'].values
tipe_transaksi_fraud = fraud_transaction['tipe_transaksi'].values
plt.hist([tipe_transaksi_fraud, tipe_transaksi_non], bins=10, label=['Fraud', 'Non-Fraud'])
plt.legend(loc='upper right')
plt.show()

kepemilikan_kartu = data['kepemilikan_kartu']
time = data['waktu_transaksi']
nama_kota = data['nama_kota']
nilai_transaksi = data['nilai_transaksi']
rata_rata_nilai_transaksi = data['rata_rata_nilai_transaksi']

time = time.tolist()
nilai_transaksi = nilai_transaksi.tolist()
rata_rata_nilai_transaksi = rata_rata_nilai_transaksi.tolist()

# change_format_time berfungsi utk mengubah format time menjadi 'detik'
def change_format_time(time):
    hour = int(time / 10000)
    minute = int((time - (hour*10000))/100)
    second = time - (hour*10000) - (minute*100)
    result = (hour*3600) + (minute*60) + second
    return result

selisih_transaksi = []

# Hitung selisih
def hitung_selisih(rata_rata, nilai_transaksi):
    result = abs(rata_rata - nilai_transaksi)
    return result

# result akan menampung format data 'time' dalam detik
new_time = []

for element in time:
    result = change_format_time(element)
    new_time.append(result)

"""
Pembagian waktu berdasarkan tingkat aktifitas :
    Jam istirahat : 22.00 - 07.59 --> Kategori 1
    Jam kerja : 08.00 - 18.59 --> Kategori 2
    Jam santai : 19.00 - 21.59 --> Kategori 3
"""

grouped_time = []

def categorize_time(time):
    if (time >= 79200) and (time <= 86399):
        grouped_time.append(1)
    elif (time <= 28799):
        grouped_time.append(1)
    elif (time >= 28800) and (time <= 68399):
        grouped_time.append(2)
    elif (time >= 68400) and (time <= 79199):
        grouped_time.append(3)
    else:
        pass

for element in new_time:
    result = categorize_time(element)
    grouped_time.append(result)

for i in range(0, len(nilai_transaksi)):
    result = abs(nilai_transaksi[i] - rata_rata_nilai_transaksi[i])
    selisih_transaksi.append(result)
    
data['grouped_time'] = [x for x in grouped_time if x is not None]
data['selisih_transaksi'] = [x for x in selisih_transaksi]

grouped_time = []
selisih_transaksi = []

# new_data = data.iloc[:, [12,17,28,29,27]].values
X = data.iloc[:, [12,17,28,29]].values
y = data.iloc[:, [27]].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
# Kita dpt menggunakan Object Inspector utk melihat parameternya
imputer = Imputer(missing_values= 'NaN', strategy = 'mean', axis = 0)
# Axis mana yang ingin kita perbaiki dengan Object Imputer
imputer = imputer.fit(X[:,:])
# Selanjutnya kita pilih axis mana yg ingin kita re-place dgn data yg baru
X[:,:] = imputer.transform(X[:,:])

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

# Mengubah kategorikal data menjadi dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
# Data yg digunakan utk test sebesar 20 %
# Nilai dari random state bebas. Tidak hrs bernilai 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Correlation matrix

corr_mat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corr_mat, vmax = .8, square = True)
plt.show()