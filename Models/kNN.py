from file_reader import read_classifier
from utils import KFold_validate_logistic, show_AUC_curve, show_confusion_matrix, kNN_graph
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
days = 2
numberOfStocks = 500

(X, y, sz) = read_classifier(n_days=days)
poly = PolynomialFeatures(1)
X = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
dummy = DummyClassifier().fit(X_train, y_train)
kf = KFold(n_splits=5)
mean_error=[]
std_error=[]
k_range = [13,14,15,16,17,18,19,20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
for K in k_range:
    print(K)
    model = KNeighborsClassifier(n_neighbors=K,weights='uniform')
    temp=[] 
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        fpr, tpr, _ = roc_curve(y[test],model.predict_proba(X[test])[:,1])
        temp.append(auc(fpr, tpr))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
kNN_graph(k_range, mean_error, std_error)
"""
y_pred = model2.predict(X_test)
y_pred_dummy = dummy.predict(X_test)
print(y_pred)
show_confusion_matrix(y_test, y_pred)
show_confusion_matrix(y_test, y_pred_dummy)
"""


#show_AUC_curve([model2, model3, model8, dummy], ["kNN-2", "kNN-3", "kNN-8", "Dummy"], X_test, y_test)
