import pandas as pd


def get_filename(type):
    size = input("Enter data size (S/M/L): ")
    if (size.lower() == "s"):
        size = "Small"
    elif (size.lower() == "m"):
        size = "Medium"
    else: size = "Large"
    return f'{type}{size}Data.csv'

def read_regression(n_days=14):
    df = pd.read_csv("../ParsedData/{}".format(get_filename("linear")), header=None)
    cols = df.shape[1]
    columns_to_remove = (14-n_days) * 13
    X = df.iloc[:, 0:(cols-2) - columns_to_remove]
    y = df.iloc[:, (cols-1)]
    print(y)
    return (X, y)


def read_classifier(n_days=14):
    df = pd.read_csv("../ParsedData/{}".format(get_filename("classifier")), header=None)
    cols = df.shape[1]
    columns_to_remove = (14-n_days) * 13
    X = df.iloc[:, 0:((cols-1) - columns_to_remove)]
    y = df.iloc[:, (cols-1)]
    return (X, y)

