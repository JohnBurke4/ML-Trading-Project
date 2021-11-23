from file_reader import read_regression
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

(X, y) = read_regression(n_days=2)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LinearRegression().fit(X_train, y_train)
dummy = DummyRegressor().fit(X_train, y_train)
print("Linear regression: ", model.score(X_test, y_test))
print("Dummy regression: ", dummy.score(X_test, y_test))
"""
for i in range(len(X_test)):
    print(f"{model.predict(X_test[i])}-{y_test[i]}")
"""
