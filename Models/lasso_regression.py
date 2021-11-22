from file_reader import read_classifier
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold


def cToAlpha(c):
    return 1 / (2*c)


filename = 'data2Days.csv'
(X, y) = read_classifier(filename=filename)
poly = PolynomialFeatures(1)
X = poly.fit_transform(X)
kf = KFold(n_splits=5)
kf.get_n_splits(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = Lasso(cToAlpha(10)).fit(X_train, y_train)
dummy = DummyClassifier().fit(X_train, y_train)
print("Linear: ", model.score(X_test, y_test))
print("DummyClassifier: ", dummy.score(X_test, y_test))
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
