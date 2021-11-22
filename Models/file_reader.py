import pandas as pd


def read_regression():
    df = pd.read_csv(f"../ParsedData/Data.csv", header=None)
    X = df.iloc[:, 0:66]
    y = df.iloc[:, 66]
    return (X, y)


def read_classifier(filename="DataClassifier.csv"):
    df = pd.read_csv(f"../ParsedData/{filename}", header=None)
    cols = df.shape[1]
    X = df.iloc[:, 0:(cols-2)]
    y = df.iloc[:, (cols-1)]
    return (X, y)
