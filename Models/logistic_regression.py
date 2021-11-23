from file_reader import read_classifier
from utils import KFold_validate
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

numberOfStocks = 500

days = 2
filename = f'dataClassifierTrain{days}Days.csv'
testFilename = f'dataClassifierTest{days}Days.csv'

(X, y) = read_classifier(filename=filename)
(X_test, y_test) = read_classifier(filename=testFilename)

poly = PolynomialFeatures(1)
X = poly.fit_transform(X)
X_test = poly.fit_transform(X_test)

#X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression(
    penalty='none', solver='lbfgs', max_iter=1000).fit(X, y)
dummy = DummyClassifier().fit(X, y)
print("LogisticRegression: ", model.score(X_test, y_test))
print("DummyClassifier: ", dummy.score(X_test, y_test))


def makeMoney(model, dummyModel, X, cash=10000):
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


makeMoney(model, dummy, X_test, cash=1000)
