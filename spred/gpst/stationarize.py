""" Script for checking if time series data is stationary. """
import numpy as np
import pandas as pd

from matplotlib import pyplot
from statsmodels.tsa.stattools import adfuller




class StationarityTests:
    """ Class for testing stationarity. """

    def __init__(self, significance=0.05):
        self.significance_level = significance
        self.p_value = None
        self.is_stationary = None

    def adf_stationarity_test(self, timeseries, print_results=True):
        """ Dickey-Fuller test. """
        adf_test = adfuller(timeseries, autolag="AIC")
        self.p_value = adf_test[1]
        if self.p_value < self.significance_level:
            self.is_stationary = True
        else:
            self.is_stationary = False
        if print_results:
            df_results = pd.Series(
                adf_test[0:4],
                index=[
                    "ADF Test Statistic",
                    "P-Value",
                    "# Lags Used",
                    "# Observations Used",
                ],
            )
            # Add Critical Values
            for key, value in adf_test[4].items():
                df_results["Critical Value (%s)" % key] = value
            print("Augmented Dickey-Fuller Test Results:")
            print(df_results)


def main() -> None:
    """ Read dataset, check stationarity, and optionally graph. """
    data = pd.read_csv("../../../ETHUSDT_ta_drop.csv", sep="\t")
    # TODO: what is ``n``?
    n = 1
    data = data.iloc[::n, :]
    print(data.head())

    # pylint: disable=invalid-name
    PLOT = True
    for col in data.columns:
        data[col] = data[col] - data[col].shift(1)
        s_test = StationarityTests()
        series = pd.Series(data[col][1:], n * np.arange(1, len(data[col])))
        series_agg = []
        k = 30
        for i in range(series.shape[0] // k):
            # print(series.iloc[i:i+k].values.sum())
            series_agg.append(series.iloc[i : i + k].values.sum())
        if PLOT:
            # print(series_agg)
            # plt_series = series_agg[:100]
            # plt_series = pd.Series(plt_series, n * k * np.arange(0, len(plt_series)))
            s_test.adf_stationarity_test(series_agg, True)
            pyplot.show()
            PLOT = False
            break
        s_test.adf_stationarity_test(series, False)
        if s_test.is_stationary:
            print("Column stationary:", col, "with p-value:", s_test.p_value)
        else:
            print("Column not stationary:", col, "with p-value:", s_test.p_value)


if __name__ == "__main__":
    main()
