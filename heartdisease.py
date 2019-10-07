# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:58:48 2019

@author: irffy
"""


import pandas as pd
from pandas.compat import StringIO
import re
import numpy as np 
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
from keras.models import Sequential
from keras.layers import Dense
import warnings


df = pd.read_csv("heart.csv")
cols = ['age', 'sex', 'cp', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'] #columns after elimation of pvalues>0.5, method is shown below
df = df[cols]
X = np.array(df.drop(['target'], 1))

y = np.array(df.target.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary2()) #elimnating variables in logistic regression with a p value > 0.05

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
print("regression accuracy:",metrics.accuracy_score(y_test, y_pred))
mlp = MLPClassifier(hidden_layer_sizes=(13)) #13 nodes 1 for each variable, single layer simple neural network
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

# using keras sequantial model
print("mlp:",metrics.accuracy_score(y_test, predictions)) 
classifier = Sequential()

classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu', input_dim = 9))


classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu'))


classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 20, nb_epoch = 100)

from sklearn import svm

from sklearn.metrics import classification_report,confusion_matrix
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel


clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
print("svm Accuracy:",metrics.accuracy_score(y_test, y_pred))
