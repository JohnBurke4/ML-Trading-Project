from file_reader import read_classifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold,ShuffleSplit
from matplotlib import pyplot as plt
from sklearn.metrics import auc, confusion_matrix, classification_report, roc_curve,f1_score
# Logisitic Regression -- different penalties 

def ratl1l2(day):
    days = day
    rat_range = [0.1,0.25,0.5,0.75,0.9]
    (X, y,sz) = read_classifier(n_days=days)
    kf = KFold(n_splits=5)
    f1_arr = []
    std_err = []
    poly = PolynomialFeatures(1)
    xpoly = poly.fit_transform(X)
    for rat in rat_range:
        temp = []
        model = LogisticRegression(
                penalty='elasticnet', solver='saga', dual=False,max_iter=10000,l1_ratio=rat)
        for train, test in kf.split(xpoly):
            model.fit(xpoly[train], y[train])
            temp.append(f1_score(y[test],model.predict(xpoly[test])))
        f1_arr.append(np.array(temp).mean())
        std_err.append(np.array(temp).std())
    plt.figure(1)
    plt.errorbar(rat_range, f1_arr, yerr=std_err, linewidth=3)
    plt.title("Elasticnet Ratio Performance")
    plt.ylabel("F1 Score")
    plt.xlabel("L1:L2 Ratio")
    plt.legend(['Logistic Regression Elasticnet'])
    plt.show()

def cvalues(day):
    q_range = [1,2]
    c_range = [0.1,1, 10, 100, 1000]
    days = day
    (X, y,sz) = read_classifier(n_days=days)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy2 = DummyClassifier(strategy="uniform")
    kf = KFold(n_splits=5)
    for q in q_range:
        auc_arr0 = []
        std_error0 = []
        auc_arr = []
        std_error = []
        auc_arr1 = []
        std_error1 = []
        auc_arr2 = []
        std_error2 = []
        auc_arr3 = []
        std_error3 = []
        auc_arr4 = []
        std_error4 = []
        poly = PolynomialFeatures(q)
        xpoly = poly.fit_transform(X)
        for c in c_range:
            temp0 = []
            temp = []
            temp1 = []
            temp2 = []
            temp3 = []
            temp4=[]
            model0 = LogisticRegression( penalty='none',solver='lbfgs',max_iter=10000)
            model = LogisticRegression(
                penalty='l1', solver='saga', dual=False,C=c,max_iter=10000)
            model1 = LogisticRegression(
                penalty='l2', solver='lbfgs', dual=False,C=c,max_iter=10000)
            model2 = LogisticRegression(
                penalty='elasticnet', solver='saga', dual=False,C=c,max_iter=10000,l1_ratio=0.5)
            for train, test in kf.split(xpoly):
                model0.fit(xpoly[train], y[train])
                model.fit(xpoly[train], y[train])
                model1.fit(xpoly[train], y[train])
                model2.fit(xpoly[train], y[train])
                dummy.fit(xpoly[train], y[train])
                dummy2.fit(xpoly[train], y[train])
                fpr, tpr, _ = roc_curve(
                    y[test], model0.decision_function(xpoly[test]))
                temp0.append(auc(fpr, tpr))
                fpr, tpr, _ = roc_curve(
                    y[test], model2.decision_function(xpoly[test]))
                temp4.append(auc(fpr, tpr))
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
            auc_arr0.append(np.array(temp0).mean())
            std_error0.append(np.array(temp0).std())
            auc_arr.append(np.array(temp).mean())
            std_error.append(np.array(temp).std())
            auc_arr1.append(np.array(temp1).mean())
            std_error1.append(np.array(temp1).std())
            auc_arr2.append(np.array(temp2).mean())
            std_error2.append(np.array(temp2).std())
            auc_arr3.append(np.array(temp3).mean())
            std_error3.append(np.array(temp3).std())
            auc_arr4.append(np.array(temp4).mean())
            std_error4.append(np.array(temp4).std())
        plt.figure(q)
        plt.errorbar(c_range, auc_arr0, yerr=std_error0, linewidth=3)
        plt.errorbar(c_range, auc_arr, yerr=std_error, linewidth=3)
        plt.errorbar(c_range, auc_arr1, yerr=std_error1, linewidth=3)
        plt.errorbar(c_range, auc_arr4, yerr=std_error4, linewidth=3)
        plt.errorbar(c_range, auc_arr2, yerr=std_error2, linewidth=3)
        plt.errorbar(c_range, auc_arr3, yerr=std_error3, linewidth=3)
        print(auc_arr0)
        print(auc_arr3)
        tstr = "AUC for C penalty weight for q: " + str(q)
        plt.title(tstr)
        plt.xlabel('c')
        plt.ylabel('AUC Score')
        plt.legend(['Logistic Regression no Penalty','Logistic Regression L1','Logistic Regression L2','Logistic Regression Elasticnet','Baseline Classifier: Most Frequent','Baseline Classifier: Uniform'])
    plt.show()

def aucGraph(c1,c2,c3,day,q):
    days = day
    (X, y,sz) = read_classifier(n_days=days)
    xpoly = PolynomialFeatures(q).fit_transform(X)
    xtrain, xtest, ytrain, ytest = train_test_split(xpoly, y, test_size=0.2)
    model0 = LogisticRegression(penalty='none', solver='lbfgs',max_iter=10000).fit(xtrain,ytrain)
    model2 = LogisticRegression(
                penalty='l1', solver='saga', dual=False,C=c1,max_iter=10000).fit(xtrain,ytrain)
    model3 = LogisticRegression(
                penalty='l2', solver='lbfgs', dual=False,C=c2,max_iter=10000).fit(xtrain,ytrain)
    model1 = LogisticRegression(
                penalty='elasticnet', solver='saga', dual=False,C=c3,max_iter=10000,l1_ratio=0.5).fit(xtrain,ytrain)
    ydummy = DummyClassifier(strategy="most_frequent").fit(xtrain,ytrain)
    ydummy2 = DummyClassifier(strategy="uniform").fit(xtrain,ytrain)
    fpr4, tpr4, _ = roc_curve(ytest,model1.decision_function(xtest))
    fpr0, tpr0, _ = roc_curve(ytest,model0.decision_function(xtest))
    fpr, tpr, _ = roc_curve(ytest,model2.decision_function(xtest))
    fpr1, tpr1, _ = roc_curve(ytest, model3.decision_function(xtest))
    y_base_scores = ydummy.predict_proba(xtest)
    fpr2,tpr2,_ = roc_curve(ytest, y_base_scores[:, 1])
    y_base_scores = ydummy2.predict_proba(xtest)
    fpr3,tpr3,_ = roc_curve(ytest, y_base_scores[:, 1])
    plt.figure(1)
    plt.title("ROC Curve")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot(fpr0,tpr0,color='pink')
    plt.plot(fpr,tpr,color='red')
    plt.plot(fpr1,tpr1,color='blue')
    plt.plot(fpr4,tpr4,color='black')
    plt.plot(fpr2,tpr2,color='green',linestyle='--')
    plt.plot(fpr3,tpr3,color='purple',linestyle='dashdot')
    plt.legend(['Logistic Regression no Penalty','Logistic Regression L1','Logistic Regression L2','Logistic Regression Elasticnet','Baseline Classifier: Most Frequent','Baseline Classifier: Uniform'])
    plt.show()

def f1score(day):
    q_range = [1,2]
    c_range = [0.1,1, 10, 100, 1000]
    days = day
    (X, y,sz) = read_classifier(n_days=days)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy2 = DummyClassifier(strategy="uniform")
    kf = KFold(n_splits=5)
    for q in q_range:
        f1_arr0 = []
        std_error0 = []
        f1_arr = []
        std_error = []
        f1_arr1 = []
        std_error1 = []
        f1_arr2 = []
        std_error2 = []
        f1_arr3 = []
        std_error3 = []
        f1_arr4 = []
        std_error4 = []
        poly = PolynomialFeatures(q)
        xpoly = poly.fit_transform(X)
        for c in c_range:
            temp0 = []
            temp = []
            temp1 = []
            temp2 = []
            temp3 = []
            temp4=[]
            model0 = LogisticRegression( penalty='none',solver='lbfgs',max_iter=10000)
            model = LogisticRegression(
                penalty='l1', solver='saga', dual=False,C=c,max_iter=10000)
            model1 = LogisticRegression(
                penalty='l2', solver='lbfgs', dual=False,C=c,max_iter=10000)
            model2 = LogisticRegression(
                penalty='elasticnet', solver='saga', dual=False,C=c,max_iter=10000,l1_ratio=0.5)
            for train, test in kf.split(xpoly):
                model0.fit(xpoly[train], y[train])
                model.fit(xpoly[train], y[train])
                model1.fit(xpoly[train], y[train])
                model2.fit(xpoly[train], y[train])
                dummy.fit(xpoly[train], y[train])
                dummy2.fit(xpoly[train], y[train])
                f1 = f1_score(
                    y[test], model0.predict(xpoly[test]))
                temp0.append(f1)
                f1 = f1_score(
                    y[test], model2.predict(xpoly[test]))
                temp4.append(f1)
                f1= f1_score(
                    y[test], model.predict(xpoly[test]))
                temp.append(f1)
                f1 = f1_score(
                    y[test], model1.predict(xpoly[test]))
                temp1.append(f1)
                f1= f1_score(
                    y[test], dummy.predict(xpoly[test]))
                temp2.append(f1)
                f1= f1_score(
                    y[test], dummy2.predict(xpoly[test]))
                temp3.append(f1)
            f1_arr0.append(np.array(temp0).mean())
            std_error0.append(np.array(temp0).std())
            f1_arr.append(np.array(temp).mean())
            std_error.append(np.array(temp).std())
            f1_arr1.append(np.array(temp1).mean())
            std_error1.append(np.array(temp1).std())
            f1_arr2.append(np.array(temp2).mean())
            std_error2.append(np.array(temp2).std())
            f1_arr3.append(np.array(temp3).mean())
            std_error3.append(np.array(temp3).std())
            f1_arr4.append(np.array(temp4).mean())
            std_error4.append(np.array(temp4).std())
        plt.figure(q)
        plt.errorbar(c_range, f1_arr0, yerr=std_error0, linewidth=3)
        plt.errorbar(c_range, f1_arr, yerr=std_error, linewidth=3)
        plt.errorbar(c_range, f1_arr1, yerr=std_error1, linewidth=3)
        plt.errorbar(c_range, f1_arr4, yerr=std_error4, linewidth=3)
        plt.errorbar(c_range, f1_arr2, yerr=std_error2, linewidth=3)
        plt.errorbar(c_range, f1_arr3, yerr=std_error3, linewidth=3)
        print(f1_arr0)
        print(f1_arr3)
        tstr = "F1 for C penalty weight for q: " + str(q)
        plt.title(tstr)
        plt.xlabel('c')
        plt.ylabel('F1 Score')
        plt.legend(['Logistic Regression no Penalty','Logistic Regression L1','Logistic Regression L2','Logistic Regression Elasticnet','Baseline Classifier: Most Frequent','Baseline Classifier: Uniform'])
    plt.show()

def reports(c1,c2,c3,day,q):
    days = day
    (X, y,sz) = read_classifier(n_days=days)
    xpoly = PolynomialFeatures(q).fit_transform(X)
    xtrain, xtest, ytrain, ytest = train_test_split(xpoly, y, test_size=0.2)
    model0 = LogisticRegression( penalty='none',solver='lbfgs',max_iter=10000).fit(xtrain,ytrain)
    model1 = LogisticRegression(
                penalty='elasticnet', solver='saga', dual=False,C=c3,max_iter=10000,l1_ratio=0.5).fit(xtrain,ytrain)
    model2 = LogisticRegression(
                penalty='l1', solver='saga', dual=False,C=c1,max_iter=10000).fit(xtrain,ytrain)
    model3 = LogisticRegression(
                penalty='l2', solver='lbfgs', dual=False,C=c2,max_iter=10000).fit(xtrain,ytrain)
    ydummy = DummyClassifier(strategy="most_frequent").fit(xtrain,ytrain)
    ydummy2 = DummyClassifier(strategy="uniform").fit(xtrain,ytrain)
    ypred0 = model0.predict(xtest)
    ypred1 = model2.predict(xtest)
    ypred2 = model3.predict(xtest)
    ypred3 = ydummy.predict(xtest)
    ypred4 = ydummy2.predict(xtest)
    ypred5 = model1.predict(xtest)
    print("Logistic Regression No Penalty")
    print(confusion_matrix(ytest,ypred0))
    print(classification_report(ytest,ypred0))
    print("L1 Penalty Logistic Regression")
    print(confusion_matrix(ytest,ypred1))
    print(classification_report(ytest,ypred1))
    print("L2 Penalty Logistic Regression")
    print(confusion_matrix(ytest,ypred2))
    print(classification_report(ytest,ypred2))
    print("Elasticnet Penalty Logistic Regression")
    print(confusion_matrix(ytest,ypred5))
    print(classification_report(ytest,ypred5))
    print("Baseline predictor: Most Frequent")
    print(confusion_matrix(ytest,ypred3))
    print(classification_report(ytest,ypred3))
    print("Baseline predictor: Uniform")
    print(confusion_matrix(ytest,ypred4))
    print(classification_report(ytest,ypred4))
    return (model0,model2,model3,model1)

#C optimal value seems to be 1?
# cvalues(2)
# aucGraph(1,1,1,2,1)
# reports(1,1,1,2,1)
# f1score(2)
# ratl1l2(2)