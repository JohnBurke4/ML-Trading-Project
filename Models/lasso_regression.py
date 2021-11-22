from file_reader import read_classifier
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def cToAlpha(c):
    return 1 / (2*c)


filename = 'data2Days.csv'
(X, y) = read_classifier(filename=filename)
xpoly = PolynomialFeatures(1)
poly = xpoly.fit_transform(X)
kf = KFold(n_splits=5)
# kf.get_n_splits(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
c_values = [5, 10, 15, 20, 30, 40, 50, 100]
mean_error = []
std_error = []
for c in c_values:
    model = Lasso(cToAlpha(c))
    temp = []
    for train, test in kf.split(poly):
        model.fit(poly[train], y[train])
        ypred = model.predict(poly[test])
        temp.append(mean_squared_error(y[test], ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
plt.errorbar(c_values, mean_error,yerr=std_error,linewidth=3)
plt.title("Test Data error for C")
plt.xlabel('c')
plt.ylabel('Mean Square error')
plt.legend(['test data'])
plt.show()

# model = Lasso(cToAlpha(10)).fit(X_train, y_train)
# dummy = DummyClassifier().fit(X_train, y_train)
# print("Linear: ", model.score(X_test, y_test))
# print("DummyClassifier: ", dummy.score(X_test, y_test))
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
