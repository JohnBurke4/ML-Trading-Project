from file_reader import read_file, get_params
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

apple_data = get_params(1)
X_train, X_test, y_train, y_test = train_test_split(apple_data[0], apple_data[1])
model = LinearRegression().fit(X_train, y_train)
dummy = DummyRegressor().fit(X_train, y_train)
print("Linear regression: ", model.score(X_test, y_test))
print("Dummy regression: ", dummy.score(X_test, y_test))