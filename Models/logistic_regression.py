from file_reader import read_classifier
from utils import KFold_validate_logistic, show_AUC_curve
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

days = 2
numberOfStocks = 500

(X, y) = read_classifier(n_days=days)
poly = PolynomialFeatures(1)
X = poly.fit_transform(X)
_, X_test, _, y_test = train_test_split(X, y)

model = LogisticRegression(
    penalty='none', solver='lbfgs', max_iter=10000)
dummy = DummyClassifier()
KFold_validate_logistic(model, dummy, X, y)
show_AUC_curve([model, dummy], ["Logistic", "Dummy"], X_test, y_test)


def makeMoney(model, dummy, X, cash=10000):
    stockCount = 0
    dummyCash = cash
    dummyStockCount = 0
    cols = X.shape[1]
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

#makeMoney(model, dummy, X_test, cash=1000)
