from file_reader import read_classifier
from utils import KFold_validate_logistic, show_AUC_curve, show_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

days = 2
numberOfStocks = 500

(X, y, sz) = read_classifier(n_days=days)
poly = PolynomialFeatures(1)
X = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)

model2 = KNeighborsClassifier(n_neighbors=2,weights='uniform').fit(X_train, y_train)
model3 = KNeighborsClassifier(n_neighbors=3,weights='uniform').fit(X_train, y_train)
model8 = KNeighborsClassifier(n_neighbors=8,weights='uniform').fit(X_train, y_train)
dummy = DummyClassifier().fit(X_train, y_train)
y_pred = model2.predict(X_test)
y_pred_dummy = dummy.predict(X_test)
print(y_pred)
show_confusion_matrix(y_test, y_pred)
show_confusion_matrix(y_test, y_pred_dummy)
#KFold_validate_logistic(model3, dummy, X, y)
show_AUC_curve([model2, model3, model8, dummy], ["kNN-2", "kNN-3", "kNN-8", "Dummy"], X_test, y_test)
