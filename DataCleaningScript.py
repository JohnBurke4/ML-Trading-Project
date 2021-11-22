from typing import cast
import numpy as np
import pandas as pd
import math
import csv
dataPoints = []

type = 'linear'

totalDays = 3
for days in range(1, totalDays):
    print("Day: ", days)
    dataPoints = []
    for files in range(1, 111):
        print("File: ", files)
        try:
            df = pd.read_csv("Data/Data{}.csv".format(days), header=1)

        except:
            continue

        for i in range(len(df)-1, days, -1):
            currentPoint = []
            currentRow = df.iloc[i, 1:].tolist()
            for j in range(i-days, i):
                row = df.iloc[j, 1:].tolist()
                currentPoint += row
            if (type == 'classifier'):
                currentPoint += [currentRow[0],
                                 (1 if currentRow[3] > currentRow[0] else -1)]
            elif(type == 'linear'):
                currentPoint += [currentRow[0], currentRow[3]]
            currentPoint = [6.2 if math.isnan(x) else x for x in currentPoint]
            dataPoints.append(currentPoint)

    with open("ParsedData/DataClassifier{}Days.csv".format(days), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(dataPoints)
