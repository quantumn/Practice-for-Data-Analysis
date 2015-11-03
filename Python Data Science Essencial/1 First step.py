# -*- coding: utf-8 -*-
"""
Created on Mon Nov 02 18:07:24 2015

@author: quantum
"""

###Chapter 1####
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
''''The dataset contains 506 house values that were sold in the suburbs of Boston'''
boston_dataset=datasets.load_boston()
X_full=boston_dataset.data
Y=boston_dataset.target
print X_full.shape()
print X_full.shape
print Y.shape
print boston_dataset.DESCR
selector=SelectKBest(f_regression,k=1)
selector.fit(X_full,Y)
X=X_full[:,selector.get_support()]
print X.shape
plt.scatter(X,Y,color='black')
regressor=LinearRegression(normalize=True)
regressor.fit(X,Y)
plt.scatter(X,Y,color='black')
plt.scatter(X,Y,color='black');plt.plot(X,regressor.predict(X),color='blue',linewidth=3)
regressor=SVR()
regressor.fit(X,Y)
plt.scatter(X,Y,color='black');plt.scatter(X,regressor.predict(X),color='blue',linewidth=3);plt.show()
regressor=RandomForestRegressor()
regressor.fit(X,Y)
plt.scatter(X,Y,color='black');plt.scatter(X,regressor.predict(X),color='blue',linewidth=3)
regressor=RandomForestRegressor()
regressor.fit(X,Y)

