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
filename = 'dataClassifierTrain2Days.csv'
testFilename = 'dataClassifierTest2Days.csv'

(X, y) = read_classifier(filename=filename)
(X_test, y_test) = read_classifier(filename=filename)

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
    for i in range(1, 20):
        for j in range((10 * i)-1, (10 * i-1)-1, -1):
            prediction = model.predict(X[j].reshape(1, -1))
            dummyPrediction = dummy.predict(X[j].reshape(1, -1))

            price = X[i, 27]
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
        cash += (stockCount * price)
        stockCount = 0
        dummyCash += (dummyStockCount * price)
        dummyStockCount = 0
    print("My money: ", (cash))
    print("Dummy's Money: ", dummyCash)


makeMoney(model, dummy, X_test, cash=1000)
