from file_reader import read_classifier
from utils import KFold_validate_logistic, show_AUC_curve
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, ShuffleSplit

def gamba(model, dummy, X, X_test, cash=10000, size=1, max_days_to_trade=100):
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
    

    dummyCash = cash
    max_days_to_trade
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
        prediction = model.predict(X[index].reshape(1, -1))[0]
        dummyPrediction = dummy.predict(X[index].reshape(1, -1))[0]

        open_price = X[index, 2]
        close_price = X[index-size, 6]
        price_delta = (close_price - open_price) / open_price
        if (prediction == 1  and cash > 0 ):
            cash = (price_delta+1) * cash
        elif (prediction == -1  and cash > 0 ):
            cash = (1-price_delta) * cash  

        if (dummyPrediction == 1  and cash > 0 ):
            dummyCash = (price_delta+1) * dummyCash 
        elif (dummyPrediction == -1  and cash > 0 ):
            dummyCash = (1-price_delta) * dummyCash   


    return (cash, dummyCash)

def run_regression(X, y, sz):

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
    model = LogisticRegression(
        penalty='none', solver='lbfgs', max_iter=10000).fit(X_train, y_train)
    dummy = DummyClassifier().fit(X_train, y_train)
    #KFold_validate_logistic(model, dummy, X, y)
    #show_AUC_curve([model, dummy], ["Logistic", "Dummy"], X_test, y_test)
    (cash, dummyCash) = gamba(model, dummy, X, test_index, cash=1000, size=sz, max_days_to_trade = 300)
    return (cash, dummyCash)


totalCash = 0
totalDummyCash = 0
days = 2
(X, y, sz) = read_classifier(n_days=days)
for i in range(500):
    (cash, dummyCash) = run_regression(X, y, sz)
    totalCash += cash
    totalDummyCash += dummyCash

print(totalCash/500, totalDummyCash/500)



