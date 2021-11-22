from file_reader import read_classifier
from utils import KFold_validate
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
filename = 'dataClassifier2Days.csv'
(X, y) = read_classifier(filename=filename)
poly = PolynomialFeatures(2)
X = poly.fit_transform(X)
model = LogisticRegression(
    penalty='none', solver='lbfgs', max_iter=1500)
dummy = DummyClassifier()
(mean_auc, std_dev) = KFold_validate(model, dummy, X, y)
print("Logistic AUC: ", mean_auc[0], " with deviation: ", std_dev[0])
print("Dummy AUC: ", mean_auc[1], " with deviation: ", std_dev[1])