# -*- coding: utf-8 -*-
"""
Created on Mon Nov 02 10:40:05 2015

@author: quantum
"""

#Feature creation
'''Sometimes, you'll find yourself in a situation where features and target variables
are not really related. In this case, you can modify the input dataset, apply linear or
nonlinear transformations that can improve the accuracy of the system, and so on.
It's a very important step of the process because it completely depends on the skills
of the data scientist, who is the one responsible for artificially changing the dataset
and shaping the input data for a better fit with the classification model.
'''


from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
cali=datasets.california_housing.fetch_california_housing() #explore California Housing Data

X=cali['data']
y=cali['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor()
regressor.fit(X_train,y_train)
y_est=regressor.predict(X_test)
print "MAE=",mean_squared_error(y_test,y_est)


 #a result of 1.15 is good, but let's strive to do better.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
regressor=KNeighborsRegressor()
regressor.fit(X_train_scaled,y_train)
y_est=regressor.predict(X_test_scaled)
print 'MAE=',mean_squared_error(y_test,y_est)
#try to add a nonlinear modification to a specific feature
non_linear_feat=5
# Creating new feature (square root of it)
# Then, it's attached to the dataset
# The operation is done for both train and test set
X_train_feat=np.sqrt(X_train[:,non_linear_feat])
X_train_new_feat.shape=(X_train_new_feat.shape[0],1)
X_train_extended=np.hstack([X_train,X_train_new_feat])

X_test_new_feat = np.sqrt(X_test[:,non_linear_feat])
X_test_new_feat.shape = (X_test_new_feat.shape[0], 1)
X_test_extended = np.hstack([X_test, X_test_new_feat])
scaler=StandardScaler()


X_train_extended_scaled=scaler.fit_transform(X_train_extended)
X_test_extended_scaled=scaler.transform(X_test_extended)
regressor=KNeighborsRegressor()
regressor.fit(X_train_extended_scaled,y_train)

X_train_extended_scaled = scaler.fit_transform(X_train_extended)
X_test_extended_scaled = scaler.transform(X_test_extended)
regressor = KNeighborsRegressor()
regressor.fit(X_train_extended_scaled, y_train)
y_est=regressor.predict(X_test_extended_scaled)
print "MAE=", mean_squared_error(y_test, y_est)




from sklearn import datasets
import numpy as np
iris=datasets.load_iris()
cov_data=np.corrcoef(iris.data.T)
print iris.feature_names
cov_data
import matplotlib.pyplot as plt
img=plt.matshow(cov_data,cmap=plt.cm.winter)
plt.colorbar(img,ticks=[-1,0,1])
#Principal Component Analysis (PCA)
from sklean.decomposition import PCA
from sklearn.decomposition import PCA
pca_2c=PCA(n_components=2)
X_pca_2c=pca_2c.fit_transform(iris.data)
X_pca_2c.shape
plt.scatter(X_pca_2c[:,0],X_pca_2c[:,1],c=iris.target,alpha=.8,edgecolor='none')
plt.show()

pca_2c.explained_variance_ratio_.sum()
pca_2c.components_
pca_2cw=PCA(n_components=2,whiten=True)
X_pca_lcw=pca_2cw.fit_transform(iris.data)
plt.scatter(X_pca_lcw[:,0],X_pca_lcw[:,1],c=iris.target,alpha0=.8,edgecolors='none')
plt.scatter(X_pca_lcw[:,0],X_pca_lcw[:,1],c=iris.target,alpha=.8,edgecolors='none')
pca_lc=PCA(n_components=1)
X_pca_lc=pca_lc.fit_transform(iris.data)
plt.scatter(X_pca_lc[:,0],np.zeros(X_pca_lc.shape),c=iris.target,alpha=0.8,edgecolors='none');plt.show()
pca_lc.explained_variance_ratio_.sum()

