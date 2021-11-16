import pandas as pd

def read_regression():
    df = pd.read_csv(f"../ParsedData/Data.csv")
    X = df.iloc[:,0:66]
    y = df.iloc[:, 66]
    return (X, y)    

def read_classifier():
    df = pd.read_csv(f"../ParsedData/DataClassifier.csv")
    X = df.iloc[:,0:66]
    y = df.iloc[:, 66]
    return (X, y)