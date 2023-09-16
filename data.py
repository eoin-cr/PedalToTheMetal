import pandas as pd

readingInterval = 2000 # each data point is a 2s recording
filename = "data3"
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
    dataPoints.append(dataPoint)
    n += 1

del dataPoints[-1]

for y, x in enumerate(dataPoints):
    print(y)
    print(x)
    x.to_csv(f"./{filename}P{y}.csv", index=False)