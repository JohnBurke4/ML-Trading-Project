from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

def cToAlpha(c):
    return 1 / (2*c)

def KFold_validate_logistic(model, dummy, X, y):
    auc_mean=[]
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
    auc_mean.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
    auc_mean.append(np.array(temp1).mean())
    std_error.append(np.array(temp1).std())
    print(f"AUC Classifier: {auc_mean[0]}, Std Deviation Classifier {std_error[0]}\n" +
        f"AUC Dummy: {auc_mean[1]}, Std Deviation Dummy {std_error[1]}")
    return (auc_mean, std_error)