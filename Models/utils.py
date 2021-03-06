from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, roc_curve, confusion_matrix, auc
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
        fpr, tpr, _ = roc_curve(y[test],model.predict_proba(X[test])[:,1])
        temp.append(auc(fpr, tpr))
        fpr, tpr, _ = roc_curve(y[test],dummy.predict_proba(X[test])[:,1])
        temp1.append(auc(fpr, tpr))
    auc_mean.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
    auc_mean.append(np.array(temp1).mean())
    std_error.append(np.array(temp1).std())
    print(f"AUC Classifier: {auc_mean[0]}, Std Deviation Classifier {std_error[0]}\n" +
        f"AUC Dummy: {auc_mean[1]}, Std Deviation Dummy {std_error[1]}")
    return (auc_mean[0], std_error[0])

def show_AUC_curve(models, labels, X_test, y_test):
    for model in models:
        fpr, tpr, _ = roc_curve(y_test,model.predict_proba(X_test)[:,1])
        plt.plot(fpr,tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='green',linestyle='--')
    plt.legend(labels)
    plt.show()

def show_confusion_matrix(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))

def kNN_graph(k_range, mean_error, std_error, hline, t="kNN"):
    plt.rcParams.update({'font.size': 14})
    plt.errorbar(k_range,mean_error,yerr=std_error,linewidth=3)
    plt.hlines(hline, k_range[0], k_range[len(k_range)-1], color="red")
    plt.legend(["Dummy most frequent", "kNN"])
    plt.xlabel('k')
    plt.ylabel(t)
    plt.title(t)
    plt.show()
