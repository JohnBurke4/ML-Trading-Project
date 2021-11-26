from file_reader import read_classifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.metrics import auc, confusion_matrix, classification_report, roc_curve

# Logisitic Regression -- different penalties 
def cvalues(day):
    q_range = [1,2]
    c_range = [1, 10, 100, 1000]
    days = day
    (X, y) = read_classifier(n_days=days)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy2 = DummyClassifier(strategy="uniform")
    kf = KFold(n_splits=5)
    for q in q_range:
        auc_arr = []
        std_error = []
        auc_arr1 = []
        std_error1 = []
        auc_arr2 = []
        std_error2 = []
        auc_arr3 = []
        std_error3 = []
        poly = PolynomialFeatures(q)
        xpoly = poly.fit_transform(X)
        for c in c_range:
            temp = []
            temp1 = []
            temp2 = []
            temp3 = []
            model = LogisticRegression(
                penalty='l1', solver='saga', dual=False,C=c,max_iter=10000)
            model1 = LogisticRegression(
                penalty='l2', solver='lbfgs', dual=False,C=c,max_iter=10000)
            for train, test in kf.split(xpoly):
                model.fit(xpoly[train], y[train])
                model1.fit(xpoly[train], y[train])
                dummy.fit(xpoly[train], y[train])
                dummy2.fit(xpoly[train], y[train])
                fpr, tpr, _ = roc_curve(
                    y[test], model.decision_function(xpoly[test]))
                temp.append(auc(fpr, tpr))
                fpr, tpr, _ = roc_curve(
                    y[test], model1.decision_function(xpoly[test]))
                temp1.append(auc(fpr, tpr))
                fpr, tpr, _ = roc_curve(
                    y[test], dummy.predict_proba(xpoly[test])[:,1])
                temp2.append(auc(fpr, tpr))
                fpr, tpr, _ = roc_curve(
                    y[test], dummy2.predict_proba(xpoly[test])[:,1])
                temp3.append(auc(fpr, tpr))
            auc_arr.append(np.array(temp).mean())
            std_error.append(np.array(temp).std())
            auc_arr1.append(np.array(temp1).mean())
            std_error1.append(np.array(temp1).std())
            auc_arr2.append(np.array(temp2).mean())
            std_error2.append(np.array(temp2).std())
            auc_arr3.append(np.array(temp3).mean())
            std_error3.append(np.array(temp3).std())
        plt.figure(q)
        plt.errorbar(c_range, auc_arr, yerr=std_error, linewidth=3)
        plt.errorbar(c_range, auc_arr1, yerr=std_error1, linewidth=3)
        plt.errorbar(c_range, auc_arr2, yerr=std_error2, linewidth=3)
        plt.errorbar(c_range, auc_arr3, yerr=std_error3, linewidth=3)
        tstr = "AUC for C penalty weight for q: " + str(q)
        plt.title(tstr)
        plt.xlabel('c')
        plt.ylabel('AUC Score')
        plt.legend(['Logistic Regression L1','Logistic Regression L2','Baseline Classifier: Most Frequent','Baseline Classifier: Uniform'])
    plt.show()

def aucGraph(c1,c2,day,q):
    days = day
    (X, y) = read_classifier(n_days=days)
    xpoly = PolynomialFeatures(q).fit_transform(X)
    xtrain, xtest, ytrain, ytest = train_test_split(xpoly, y, test_size=0.2)
    logModel = LogisticRegression(
                penalty='l1', solver='saga', dual=False,C=c1,max_iter=10000).fit(xtrain,ytrain)
    logmodel2 = LogisticRegression(
                penalty='l2', solver='lbfgs', dual=False,C=c2,max_iter=10000).fit(xtrain,ytrain)
    ydummy = DummyClassifier(strategy="most_frequent").fit(xtrain,ytrain)
    ydummy2 = DummyClassifier(strategy="uniform").fit(xtrain,ytrain)
    fpr, tpr, _ = roc_curve(ytest,logModel.decision_function(xtest))
    fpr1, tpr1, _ = roc_curve(ytest, logmodel2.decision_function(xtest))
    y_base_scores = ydummy.predict_proba(xtest)
    fpr2,tpr2,_ = roc_curve(ytest, y_base_scores[:, 1])
    y_base_scores = ydummy2.predict_proba(xtest)
    fpr3,tpr3,_ = roc_curve(ytest, y_base_scores[:, 1])
    plt.figure(1)
    plt.title("Postive rate graph")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot(fpr,tpr,color='red')
    plt.plot(fpr1,tpr1,color='blue')
    plt.plot(fpr2,tpr2,color='green',linestyle='--')
    plt.plot(fpr3,tpr3,color='purple',linestyle='dashdot')
    plt.legend(['Logistic Regression L1','Logistic Regression L2','Baseline Classifier: Most Frequent','Baseline Classifier: Uniform'])
    plt.show()

def reports(c1,c2,day,q):
    days = day
    (X, y) = read_classifier(n_days=days)
    xpoly = PolynomialFeatures(q).fit_transform(X)
    xtrain, xtest, ytrain, ytest = train_test_split(xpoly, y, test_size=0.2)
    logModel = LogisticRegression(
                penalty='l1', solver='saga', dual=False,C=c1,max_iter=10000).fit(xtrain,ytrain)
    logmodel2 = LogisticRegression(
                penalty='l2', solver='lbfgs', dual=False,C=c2,max_iter=10000).fit(xtrain,ytrain)
    ydummy = DummyClassifier(strategy="most_frequent").fit(xtrain,ytrain)
    ydummy2 = DummyClassifier(strategy="uniform").fit(xtrain,ytrain)
    ypred1 = logModel.predict(xtest)
    ypred2 = logmodel2.predict(xtest)
    ypred3 = ydummy.predict(xtest)
    ypred4 = ydummy2.predict(xtest)
    print("L1 Penalty Logistic Regression")
    print(confusion_matrix(ytest,ypred1))
    print(classification_report(ytest,ypred1))
    print("L2 Penalty Logistic Regression")
    print(confusion_matrix(ytest,ypred2))
    print(classification_report(ytest,ypred2))
    print("Baseline predictor: Most Frequent")
    print(confusion_matrix(ytest,ypred3))
    print(classification_report(ytest,ypred3))
    print("Baseline predictor: Uniform")
    print(confusion_matrix(ytest,ypred4))
    print(classification_report(ytest,ypred4))

#C optimal value seems to be 1?
#cvalues(2)
aucGraph(1,1,2,1)
# reports(1,1,2,1)
