from file_reader import read_regression
from sklearn import linear_model
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def cToAlpha(c):
    return 1/(2*c)


C = 1
filename = 'dataClassifierTrain5Days.csv'
(X, y) = read_regression()


X_train, X_test, y_train, y_test = train_test_split(X, y)
model = linear_model.Lasso(alpha=cToAlpha(C)).fit(X_train, y_train)
dummy = DummyRegressor().fit(X_train, y_train)
print("LogisticRegression: ", model.score(X_test, y_test))
print("DummyClassifier: ", dummy.score(X_test, y_test))
print(model.coef_)
