from typing import cast
import numpy as np
import pandas as pd
import math
import csv

# For if we want to get data for a classifier or for a linear regression model
type = 'linear'

# The maximum days back we will use
days = 14
totalData = []
for files in range(1, 111):
    print("File: ", files)
    try:
        df = pd.read_csv("Data/Data{}.csv".format(files), header=1)
    except:
        print("Cannot find file: ", "Data/Data{}.csv".format(files))
        continue

    # Keeps track of the current points time stamp
    count = 0

    # We go from the most recent stock info, to the least recent - the number of days back we need,
    # as we dont have all the data on the least recent n days where n is the number of days back we
    # are looking at

    for i in range(len(df)-1, days-1, -1):

        currentPoint = []
        currentRow = df.iloc[i, :].tolist()
        currentPoint += [currentRow[0], currentRow[1]]

        # Adding all the stock info for the previous n days to the current data point,
        # starting from the previous day to the current day we are looking at
        for j in range(i-1, i-days-1, -1):
            row = df.iloc[j, 1:].tolist()
            currentPoint += row

        if (type == 'classifier'):
            currentPoint += [(1 if currentRow[4] > currentRow[1] else -1)]
        elif(type == 'linear'):
            currentPoint += [currentRow[4] - currentRow[1]]

        # This line replaces the NaN points corrosponding to when the US Inflation rate is unknown and uses the current months estimation
        currentPoint = [6.2 if math.isnan(x) else x for x in currentPoint]

        if (len(totalData) <= count):
            # Create a new list corrosponding to all data points that have the same timestamp
            totalData.append([currentPoint])
        else:
            # Add this data point to the list of all data points with the current time stamp
            totalData[count].append(currentPoint)

        count += 1

flattened = []
# Flattening the data list (Going from a 2d list to a 1d)
for i in range(0, len(totalData)):
    for j in range(0, len(totalData[i])):
        flattened.append(totalData[i][j])

# Writing the data to a file
with open("ParsedData/{}LargeData.csv".format(type), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(flattened)
