from file_reader import read_classifier
from utils import KFold_validate_logistic, show_AUC_curve
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, ShuffleSplit

days = 2
numberOfStocks = 500

(X, y) = read_classifier(n_days=days)
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


def makeMoney(model, dummy, X, cash=10000):
    stockCount = 0
    dummyCash = cash
    dummyStockCount = 0
    cols = X.shape[0]
    print(cols)
    price = 1
    index = 0
    #print(X[998, days*13 + 1])
    proba = 0
    for i in range(1, 11):
        #print((10 * i)-1, (10 * i-1)-1)
        for j in range((10 * i)-1, (10 * (i-1))-1, -1):
            prediction = model.predict(X[j].reshape(1, -1))
            dummyPrediction = dummy.predict(X[j].reshape(1, -1))
            tempo = model.predict_proba(X[j].reshape(1, -1))[0][1]
            if (tempo > proba):
                proba = tempo
                index = j
            # print(j)

            price = X[j, days*13 + 1]
            print(price)
            if (prediction == 1):
                if (cash > price):
                    stockCount += cash / price
                    cash -= stockCount * price
            else:
                if (stockCount != 0):
                    cash += stockCount * price
                    stockCount = 0

            if (dummyPrediction == 1):
                if (dummyCash > price):
                    dummyStockCount += dummyCash / price
                    dummyCash -= dummyStockCount * price
            else:
                if (stockCount != 0):
                    dummyCash += dummyStockCount * price
                    dummyStockCount = 0
            #print("Stock Count: ", stockCount, "Price", price)
        # print(cash)
        #print(stockCount, price)
        cash += (stockCount * price)
        # print(cash)
        stockCount = 0
        dummyCash += (dummyStockCount * price)
        dummyStockCount = 0
    print("My money: ", (cash))
    print("Dummy's Money: ", dummyCash)
    print(proba)
    print(index)

def gamba(model, dummy, X, X_test, cash=10000):
    dummyCash = cash
    rows = X.shape[0]

    price = 1
    index = 0
    #print(X[998, days*13 + 1])
    for x_t in X_test:
        if(x_t == 0):
            continue
        prediction = model.predict(X[x_t].reshape(1, -1))[0]
        dummyPrediction = dummy.predict(X[x_t].reshape(1, -1))[0]

        open_price = X[x_t, 2]
        close_price = X[x_t-1, 6]
        price_delta = (close_price - open_price) / 100
        print("delta ", price_delta)
        if (prediction == 1  and cash > 0 ):
            cash = (price_delta+1) * cash
        elif (prediction == -1  and cash > 0 ):
            cash = (1-price_delta) * cash  

        if (dummyPrediction == 1  and cash > 0 ):
            dummyCash = (price_delta+1) * dummyCash 
        elif (dummyPrediction == -1  and cash > 0 ):
            dummyCash = (1-price_delta) * dummyCash   

        print(cash, dummyCash)  

    print("My money: ", (cash))
    print("Dummy's Money: ", dummyCash)


gamba(model, dummy, X, test_index, cash=1000)
