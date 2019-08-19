import pandas as pd
import numpy as np
from matplotlib import pyplot
import statsmodels
from statsmodels.tsa.stattools import adfuller


class StationarityTests:
    def __init__(self, significance=0.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None

    def ADF_Stationarity_Test(self, timeseries, printResults=True):

        # Dickey-Fuller test:
        adfTest = adfuller(timeseries, autolag="AIC")

        self.pValue = adfTest[1]

        if self.pValue < self.SignificanceLevel:
            self.isStationary = True
        else:
            self.isStationary = False

        if printResults:
            dfResults = pd.Series(
                adfTest[0:4],
                index=[
                    "ADF Test Statistic",
                    "P-Value",
                    "# Lags Used",
                    "# Observations Used",
                ],
            )

            # Add Critical Values
            for key, value in adfTest[4].items():
                dfResults["Critical Value (%s)" % key] = value

            print("Augmented Dickey-Fuller Test Results:")
            print(dfResults)


data = pd.read_csv("../../../ETHUSDT_ta_drop.csv", sep="\t")
n = 100
data = data.iloc[::n, :]
print(data.head())

PLOT = False
for col in data.columns:
    data[col] = np.cbrt(data[col]) - np.cbrt(data[col]).shift(1)
    sTest = StationarityTests()
    series = pd.Series(data[col][1:], n * np.arange(1, len(data[col])))
    if PLOT:
        print(series)
        series.plot()
        pyplot.show()
        PLOT = False
    sTest.ADF_Stationarity_Test(series, False)
    if sTest.isStationary:
        print("Column stationary:", col, "with p-value:", sTest.pValue)
    else:
        print("Column not stationary:", col, "with p-value:", sTest.pValue)