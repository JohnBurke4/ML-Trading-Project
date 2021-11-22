from file_reader import read_classifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
filename = 'dataClassifier2Days.csv'
(X, y) = read_classifier(filename=filename)
poly = PolynomialFeatures(2)
X = poly.fit_transform(X)
kf = KFold(n_splits=5)
kf.get_n_splits(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression(
    penalty='none', solver='lbfgs', max_iter=1000).fit(X_train, y_train)
dummy = DummyClassifier().fit(X_train, y_train)
print("LogisticRegression: ", model.score(X_test, y_test))
print("DummyClassifier: ", dummy.score(X_test, y_test))
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
