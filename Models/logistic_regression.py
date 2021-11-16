from file_reader import read_classifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

(X, y) = read_classifier()
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LogisticRegression(penalty='none',solver='lbfgs', max_iter=1000).fit(X_train, y_train)
dummy = DummyClassifier().fit(X_train, y_train)
print("LogisticRegression: ", model.score(X_test, y_test))
print("DummyClassifier: ", dummy.score(X_test, y_test))
"""
for i in range(len(X_test)):
    print(f"{model.predict(X_test[i])}-{y_test[i]}")
"""