# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 23:26:56 2018

@author: Gerry
"""

from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

class NN():
    def __init__(self, depth = 1, hidden_node = 32, input_node = int, dropout = 0, output_node = int):
        self._depth = depth
        self._hidden_node = hidden_node
        self._input_node = input_node
        self._dropout = dropout
        self._output_node = output_node
        
    def ann(self):        
        # initialising the ANN
        classifier = Sequential()        
        classifier.add(Dense(output_dim = self._hidden_node, init = 'uniform', activation= 'relu', input_dim = self._input_node))
        classifier.add(Dropout(p = self._dropout))
        if self._depth > 1:
            for i in range (self._depth - 1):
                classifier.add(Dense(output_dim = self._hidden_node, init = 'uniform', activation= 'relu'))
                classifier.add(Dropout(p = self._dropout))
        classifier.add(Dense(output_dim = self._output_node, init = 'uniform', activation= 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier
    
    def ann_cv(self, batch, epoch, X_train, y_train, cv):
        classifier = KerasClassifier(build_fn = self.ann, batch_size = batch, epochs = epoch )
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = cv)
        return accuracies
               
    def cnn(self, n_feature = 32, f_size = 3, pool_size = 2):
        classifier = Sequential()
        classifier.add(Convolution1D(n_feature, f_size, input_shape = (self._input_node, 1), activation = 'relu'))
        classifier.add(MaxPooling1D(pool_size = (pool_size)))
        classifier.add(Flatten())
        
        for i in range (self._depth):
            classifier.add(Dense(output_dim = self._hidden_node, init = 'uniform', activation= 'relu'))
            classifier.add(Dropout(p = self._dropout))
        classifier.add(Dense(output_dim = self._output_node, init = 'uniform', activation= 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier
    
