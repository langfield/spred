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
n = 1
data = data.iloc[::n, :]
print(data.head())

PLOT = True
for col in data.columns:
    data[col] = data[col] - data[col].shift(1)
    sTest = StationarityTests()
    series = pd.Series(data[col][1:], n * np.arange(1, len(data[col])))
    series_agg = []
    k = 30
    for i in range(series.shape[0] // k):
        # print(series.iloc[i:i+k].values.sum())
        series_agg.append(series.iloc[i:i+k].values.sum())
    
    if PLOT:
        # print(series_agg)
        plt_series = series_agg[:100]
        #plt_series = pd.Series(plt_series, n * k * np.arange(0, len(plt_series)))
        sTest.ADF_Stationarity_Test(series_agg, True)
        pyplot.show()
        PLOT = False
        break
    sTest.ADF_Stationarity_Test(series, False)
    if sTest.isStationary:
        print("Column stationary:", col, "with p-value:", sTest.pValue)
    else:
        print("Column not stationary:", col, "with p-value:", sTest.pValue)
