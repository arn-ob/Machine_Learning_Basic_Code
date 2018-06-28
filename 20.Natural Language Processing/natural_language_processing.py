# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 00:33:40 2018

Natural Language Processing

@author: Arnob
"""

#### import the lib
import numpy as np
import matplotlib as plt
import pandas as pd

#### import the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#### cleaning the texts
import re

# for downloading
# import nltk
# nltk.download('stopwords') 

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []  # in NLP corpus is collection of data
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' ' .join(review)
    corpus.append(review)


#### Createing the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

 # From Native bayes classification
# Splitting the dataset into tranning set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)

# Fitting Native Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Prediction the test set result
y_pred = classifier.predict(X_test)


# Making the cofusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Accuracy Calculation
# (55+91)/200
accuracy = (cm[0][0] + cm[1][1])/(cm[0][0]+ cm[0][1] + cm[1][0]+ cm[1][1]) * 100