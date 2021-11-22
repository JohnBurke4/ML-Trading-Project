from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

def cToAlpha(c):
    return 1 / (2*c)

def KFold_validate(model, dummy, X, y):
    mean_error=[]
    std_error=[]
    kf = KFold(n_splits=5)
    temp = []
    temp1 = []
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        dummy.fit(X[train], y[train])
        ypred = model.predict(X[test])
        ypred_dummy = dummy.predict(X[test])
        temp.append(roc_auc_score(y[test], ypred))
        temp1.append(roc_auc_score(y[test], ypred_dummy))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
    mean_error.append(np.array(temp1).mean())
    std_error.append(np.array(temp1).std())
    return (mean_error, std_error)