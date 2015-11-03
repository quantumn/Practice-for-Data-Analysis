#Linear Regrssion
from sklearn.datasets import load_boston
boston=load_boston()
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(boston.data,boston.target,test_size=0.2,													random_state=0)


from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
from sklearn.metrics import mean_absolute_error
print "MAE", mean_absolute_error(y_test, y_pred)
%timeit regr.fit(X_train, y_train)