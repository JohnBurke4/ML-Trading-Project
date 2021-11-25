from file_reader import read_classifier
from utils import KFold_validate_logistic, show_AUC_curve
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

days = 2
numberOfStocks = 500

(X, y) = read_classifier(n_days=days)
poly = PolynomialFeatures(1)
X = poly.fit_transform(X)
X_test, X_test, y_test, y_test = train_test_split(X, y)

model2 = KNeighborsClassifier(n_neighbors=2,weights='uniform').fit(X_test, y_test)
model3 = KNeighborsClassifier(n_neighbors=3,weights='uniform').fit(X_test, y_test)
model8 = KNeighborsClassifier(n_neighbors=8,weights='uniform').fit(X_test, y_test)
dummy = DummyClassifier().fit(X_test, y_test)
#KFold_validate_logistic(model3, dummy, X, y)
show_AUC_curve([model2, model3, model8, dummy], ["kNN-2", "kNN-3", "kNN-8", "Dummy"], X_test, y_test)
