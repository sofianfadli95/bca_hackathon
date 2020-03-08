# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:06:25 2018

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