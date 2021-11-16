import numpy as np
import pandas as pd

"""
@params: data_id: The file id to parse
"""


def read_file(data_id):
    # time,open,high,low,close,MA,Volume,Volume MA,Plot,Gross profit,Free cash flow,RSI,MOM,U.S. Inflation
    d = dict()
    df = pd.read_csv(f"../Data/Data{data_id}.csv")
    d['time'] = df.iloc[:, 0]
    d['open'] = df.iloc[:, 1]
    d['high'] = df.iloc[:, 2]
    d['low'] = df.iloc[:, 3]
    d['close'] = df.iloc[:, 4]
    d['MA'] = df.iloc[:, 5]
    d['volume'] = df.iloc[:, 6]
    d['volume MA'] = df.iloc[:, 7]
    d['sentiment'] = df.iloc[:, 8]
    d['gross profit'] = df.iloc[:, 9]
    d['free cash flow'] = df.iloc[:, 10]
    d['RSI'] = df.iloc[:, 11]
    d['momentum'] = df.iloc[:, 12]
    d['inflation'] = df.iloc[:, 13]
    return d


"""
@params: data_id: The file id to parse
        data_range: The amount of days to take into account before the prediction
"""
def get_params(data_id, data_range=5):
    data = read_file(data_id)
    y = []
    X_open1 = []
    X_open2 = []
    X_open3 = []
    X_open4 = []
    X_open5 = []
    X_open6 = []
    X_high1 = []
    X_high2 = []
    X_high3 = []
    X_high4 = []
    X_high5 = []
    X_low1 = []
    X_low2 = []
    X_low3 = []
    X_low4 = []
    X_low5 = []
    X_close1 = []
    X_close2 = []
    X_close3 = []
    X_close4 = []
    X_close5 = []
    X_MA1 = []
    X_MA2 = []
    X_MA3 = []
    X_MA4 = []
    X_MA5 = []
    X_RSI1 = []
    X_RSI2 = []
    X_RSI3 = []
    X_RSI4 = []
    X_RSI5 = []
    X_sentiment1 = []
    X_sentiment2 = []
    X_sentiment3 = []
    X_sentiment4 = []
    X_sentiment5 = []
    length = data["time"].size
    for i in range(length - 5):
        X_open1.append(data["open"][i])
        X_open2.append(data["open"][i+1])
        X_open3.append(data["open"][i+2])
        X_open4.append(data["open"][i+3])
        X_open5.append(data["open"][i+4])
        X_open6.append(data["open"][i+5])
        X_high1.append(data["high"][i])
        X_high2.append(data["high"][i+1])
        X_high3.append(data["high"][i+2])
        X_high4.append(data["high"][i+3])
        X_high5.append(data["high"][i+4])
        X_low1.append(data["low"][i])
        X_low2.append(data["low"][i+1])
        X_low3.append(data["low"][i+2])
        X_low4.append(data["low"][i+3])
        X_low5.append(data["low"][i+4])
        X_close1.append(data["close"][i])
        X_close2.append(data["close"][i+1])
        X_close3.append(data["close"][i+2])
        X_close4.append(data["close"][i+3])
        X_close5.append(data["close"][i+4])
        X_MA1.append(data["MA"][i])
        X_MA2.append(data["MA"][i+1])
        X_MA3.append(data["MA"][i+2])
        X_MA4.append(data["MA"][i+3])
        X_MA5.append(data["MA"][i+4])
        X_RSI1.append(data["RSI"][i])
        X_RSI2.append(data["RSI"][i+1])
        X_RSI3.append(data["RSI"][i+2])
        X_RSI4.append(data["RSI"][i+3])
        X_RSI5.append(data["RSI"][i+4])
        X_sentiment1.append(data["sentiment"][i])
        X_sentiment2.append(data["sentiment"][i+1])
        X_sentiment3.append(data["sentiment"][i+2])
        X_sentiment4.append(data["sentiment"][i+3])
        X_sentiment5.append(data["sentiment"][i+4])
        y.append(data["close"][i+5])


    X = np.column_stack((X_open1, X_open2, X_open3, X_open4, X_open5, X_open6, X_high1, X_high2, X_high3, X_high4, X_high5,
                     X_low1, X_low2, X_low3, X_low4, X_low5, X_close1, X_close2, X_close3, X_close4, X_close5, X_MA1, 
                     X_MA2, X_MA3, X_MA4, X_MA5, X_RSI1, X_RSI2, X_RSI3, X_RSI4, X_RSI5, X_sentiment1, X_sentiment2, X_sentiment3, 
                     X_sentiment4, X_sentiment5))
    return (X, y)
