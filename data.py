import pandas as pd
import os

readingInterval = 2000 # each data point is a 2s recording

def parseFile(filename):
    if filename.endswith(".csv"):
        filename = filename[:-4]
    if not os.path.exists(f"./{filename}"):
        os.mkdir(f"./{filename}")
    data = pd.read_csv(f"./{filename}.csv")

    data = data.apply(pd.to_numeric, errors='coerce')
    data["time"] = data["time"].astype("int64")

    firstTime = data["time"].values[0]

    data["timeOffset"] = data["time"] - firstTime

    dataPoints = []

    n = 0
    while True:
        dataPoint = data[n*readingInterval <= data["timeOffset"]]
        dataPoint = dataPoint[dataPoint["timeOffset"] < (n+1)*readingInterval]
        if len(dataPoint.index) <= 0: # stop when there are no more lines
            break
        firstTimeDataPoint = dataPoint["time"].values[0]
        dataPoint["timeOffset"] = dataPoint["time"] - firstTimeDataPoint
        dataPoints.append(dataPoint)
        n += 1

    del dataPoints[-1]

    for y, x in enumerate(dataPoints):
        print(y)
        print(x)
        x.to_csv(f"./{filename}/{filename.split('/')[-1]}-Parsed{y}.csv", index=False)

mode = input("file or folder? ")
if mode == "file":
    parseFile(input("Filename/path: "))
elif mode  == "folder":
    dirname = input("Folder name/path: ")
    for filename in os.listdir(dirname):
        fn = os.path.join(dirname, filename)
        parseFile(fn)