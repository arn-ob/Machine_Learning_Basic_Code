# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 22:46:22 2018

Artificial Neural Network

@author: Arnob
"""

####################################
######### Part 1: Data Preprocessing
####################################

# import lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values   # indipendent Variable
y = dataset.iloc[:, 13].values     # depended Variable 

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Convert String to identical number
# This for country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# This for gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]


# Splitting the dataset into tranning set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

####################################
######### Part 2: Making and Processing ANN
####################################

# import lib
# import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the Artificial Neural Network
classifier = Sequential()

# Adding input layer, 1st hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) 

# Adding input layer, 2nd hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu')) 

# Adding output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid')) 

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Prediction the test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the cofusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)