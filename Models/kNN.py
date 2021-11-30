from file_reader import read_classifier
from utils import show_AUC_curve, show_confusion_matrix, kNN_graph, KFold_validate_logistic
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc,f1_score
days = 2
numberOfStocks = 500

(X, y, sz) = read_classifier(n_days=days)
poly = PolynomialFeatures(1)
X = poly.fit_transform(X)
kf = KFold(n_splits=5)
X_train, X_test, y_train, y_test = train_test_split(X, y)
"""
dummy_temp = []
dummy_mean_error = []
dummy_std_error = []
dummy = DummyClassifier(strategy="most_frequent")
for train, test in kf.split(X):
    dummy.fit(X[train], y[train])
    f1 = f1_score(y[test],dummy.predict(X[test]))
    dummy_temp.append(f1)
dummy_mean_error = np.array(dummy_temp).mean()
dummy_std_error = np.array(dummy_temp).std()
mean_error = []
std_error = []
k_range = list(range(1, 50))
for K in k_range:
    print(K)
    model = KNeighborsClassifier(n_neighbors=K,weights='uniform')
    temp=[] 
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        f1 = f1_score(y[test], y_pred)
        temp.append(f1)
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
kNN_graph(k_range, mean_error, std_error, dummy_mean_error, "F1 Score")
print(max(mean_error), mean_error.index(max(mean_error))+1)
"""

model = KNeighborsClassifier(n_neighbors=2,weights='uniform').fit(X_train, y_train)
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
dummy2 = DummyClassifier(strategy="uniform").fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_dummy = dummy.predict(X_test)
y_pred_dummy2 = dummy2.predict(X_test)
show_confusion_matrix(y_test, y_pred)
show_confusion_matrix(y_test, y_pred_dummy)
show_confusion_matrix(y_test, y_pred_dummy2)
"""
models_names = ["logistic", "l1", "l2", "kNN"]
models = []
models.append(LogisticRegression(solver="lbfgs", max_iter=10000))
models.append(LogisticRegression(penalty='l1', solver='saga', dual=False,C=1,max_iter=10000))
models.append(LogisticRegression(penalty='l2', solver='lbfgs', dual=False,C=1, max_iter=10000))
models.append(KNeighborsClassifier(n_neighbors=42,weights='uniform'))
dummy = DummyClassifier(strategy="most_frequent").fit(X[train], y[train])
auc_means =[]
errorbars=[] 
for m in models:      
    (auc, error) = KFold_validate_logistic(m, dummy, X, y)
    auc_means.append(auc)
    errorbars.append(error)
kNN_graph(models_names, auc_means, errorbars, 0.5)
models_names.append("Dummy")
models.append(dummy)
show_AUC_curve(models, models_names, X_test, y_test)
"""