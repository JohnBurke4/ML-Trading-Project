import pandas as pd


def read_regression(filename="Data.csv"):
    df = pd.read_csv("../ParsedData/{}".format(filename), header=None)
    cols = df.shape[1]
    X = df.iloc[:, 0:(cols-2)]
    y = df.iloc[:, (cols-1)]
    return (X, y)


def read_classifier(filename="DataClassifier.csv"):
    df = pd.read_csv("../ParsedData/{}".format(filename), header=None)
    cols = df.shape[1]
    X = df.iloc[:, 0:(cols-1)]
    y = df.iloc[:, (cols-1)]
    return (X, y)
