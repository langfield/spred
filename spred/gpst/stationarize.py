""" Checks if time series data is stationary. """
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller

# pylint: disable=too-few-public-methods
class StationarityTests:
    """ Class for testing stationarity. """

    def __init__(self, significance: float = 0.05) -> None:
        self.significance_level = significance
        self.p_value = None
        self.is_stationary = False

    def adf_stationarity_test(self, series: pd.Series, debug: bool = True) -> None:
        """
        Dickey-Fuller test.

        Parameters
        ----------
        series : ``pd.Series``, required.
            The time series to test for stationarity.
        debug : ``bool``.
            Whether or not to print statistics.
        """
        adf_test = adfuller(series, autolag="AIC")
        self.p_value = adf_test[1]
        if self.p_value < self.significance_level:
            self.is_stationary = True
        else:
            self.is_stationary = False
        if debug:
            df_results = pd.Series(
                adf_test[0:4],
                index=[
                    "ADF Test Statistic",
                    "P-Value",
                    "# Lags Used",
                    "# Observations Used",
                ],
            )
            # Add Critical Values.
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

    for col in data.columns:
        data[col] = data[col] - data[col].shift(1)
        s_test = StationarityTests()
        series = pd.Series(data[col][1:], n * np.arange(1, len(data[col])))
        s_test.adf_stationarity_test(series, False)
        if s_test.is_stationary:
            print("Column stationary:", col, "with p-value:", s_test.p_value)
        else:
            print("Column not stationary:", col, "with p-value:", s_test.p_value)


if __name__ == "__main__":
    main()
