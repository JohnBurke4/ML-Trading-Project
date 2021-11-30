from file_reader import read_classifier
from utils import KFold_validate_logistic, show_AUC_curve, show_confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

def gamba(model, X, X_test, cash=10000, size=1, max_days_to_trade=100):
    X_test = np.sort(X_test)
    X_test_day = []
    cap = size
    temp_arr = []
    for x_t in X_test:
        if(x_t < cap):
            temp_arr.append(x_t)
        else:
            cap = cap + size
            X_test_day.append(temp_arr)
            temp_arr = [x_t]
    if(len(temp_arr) > 0):
        X_test_day.append(temp_arr)
    
    count = 0
    for x_t in X_test_day:
        count = count + 1
        if(count > max_days_to_trade):
            break
        index = 0
        buy = 1
        em = 0
        for x_d in x_t:
            
            model_proba = model.predict_proba(X[x_d].reshape(1, -1))[0]
            if(model_proba[0] > em and model_proba[0] > model_proba[1]):
                index = x_d
                buy = -1
                em = model_proba[0]
            elif(model_proba[0] > em and model_proba[1] > model_proba[0]):
                index = x_d
                buy = 1
                em = model_proba[1]

        if(index < size):
            continue
        prediction = buy

        open_price = X[index, 2]
        close_price = X[index-size, 6]
        price_delta = (close_price - open_price) / open_price
        if (prediction == 1  and cash > 0 ):
            cash = (price_delta+1) * cash
        elif (prediction == -1  and cash > 0 ):
            cash = (1-price_delta) * cash  


    return cash

def run_regression(X, y, sz, model):

    poly = PolynomialFeatures(1)
    X = poly.fit_transform(X)
    sss = ShuffleSplit(n_splits=1, test_size=0.2)

    data_size = X.shape[0]
    cols = X.shape
    X_indices = np.reshape(np.random.rand(data_size*2),(data_size,2))
    y_indices = np.random.randint(2, size=data_size)

    sss.get_n_splits(X, y)
    train_index, test_index = next(sss.split(X_indices, y_indices)) 

    X_train, X_test = X[train_index], X[test_index] 
    y_train, y_test = y[train_index], y[test_index]
    #the close price on a test day
    model.fit(X_train, y_train)
    
    cash = gamba(model, X, test_index, cash=1000, size=sz, max_days_to_trade = 300)
    return cash

def trade():
    model_names = ["logistic", "kNN-19", "most frequent"]
    models = []
    saved_cash = []
    saved_std = []
    models.append(LogisticRegression(solver="lbfgs", max_iter=10000))
    models.append(LogisticRegression(penalty='l2', solver='lbfgs', dual=False,C=1,max_iter=10000))
    #models.append(KNeighborsClassifier(n_neighbors=2,weights='uniform'))
    #models.append(DummyClassifier(strategy="uniform"))
    days = 2
    runs = 2500
    (X, y, sz) = read_classifier(n_days=days)
    for model in models:  
        totalCash = []
        for i in range(runs):
            print(i)
            cash = run_regression(X, y, sz, model)
            totalCash.append(cash)
        saved_cash.append(np.array(totalCash).mean())
        saved_std.append(np.array(totalCash).std())
    for i in range (len(saved_cash)):
        print(f"{model_names[i]} cash: {saved_cash[i]} std: {saved_std[i]}")

def regression():
    days = 2
    (X, y, sz) = read_classifier(n_days=days)
    run_regression(X, y, sz)


trade()

