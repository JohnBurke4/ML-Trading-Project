import pandas as pd


def get_filename(type):
    n_stocks = 1
    size = input("Enter data size (S/M/L): ")
    if (size.lower() == "s"):
        size = "Small"
    elif (size.lower() == "m"):
        size = "Medium"
        n_stocks = 10
    else: 
        size = "Large"
        n_stocks = 100

    return (f'{type}{size}Data.csv', n_stocks)

def read_regression(n_days=14):
    (filename, n_stocks) = get_filename("linear")
    df = pd.read_csv("../ParsedData/{}".format(filename), header=None)
    cols = df.shape[1]
    columns_to_remove = (14-n_days) * 13
    X = df.iloc[:, 0:(cols-2) - columns_to_remove]
    y = df.iloc[:, (cols-1)]
    print(y)
    return (X, y, n_stocks)


def read_classifier(n_days=14):
    (filename, n_stocks) = get_filename("classifier")
    df = pd.read_csv("../ParsedData/{}".format(filename), header=None)
    cols = df.shape[1]
    columns_to_remove = (14-n_days) * 13
    X = df.iloc[:, 0:((cols-1) - columns_to_remove)]
    y = df.iloc[:, (cols-1)]
    return (X, y, n_stocks)

