from typing import cast
import numpy as np
import pandas as pd
import math
import csv
dataPoints = []
for k in range(1, 111):
    print(k)
    try:
        df = pd.read_csv("Data/Data{}.csv".format(k), header=1)

    except:
        continue

    for i in range(len(df)-1, 5, -1):
        currentPoint = []
        currentRow = df.iloc[i, 1:].tolist()
        for j in range(i-5, i):
            row = df.iloc[j, 1:].tolist()
            currentPoint += row
        currentPoint += [currentRow[0], currentRow[3]]
        currentPoint = [6.2 if math.isnan(x) else x for x in currentPoint]
        dataPoints.append(currentPoint)

with open("ParsedData/Data.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dataPoints)
