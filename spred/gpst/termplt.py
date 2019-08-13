""" Plots a ``.csv`` file to terminal. """
import os
import platform
import numpy as np
import pandas as pd
from terminalplot import plot


def plot_to_terminal(series: np.ndarray) -> None:
    """ Plots the first dimension of a ``np.ndarray`` to the terminal. """
    assert platform.system() == "Linux"
    try:
        y = list(series[:, 0])  # Assumes 2-dimensional data.
    except IndexError:
        y = list(series) # 1-dimensional data.
    x = [i for i in range(series.shape[0])]
    plot(x, y)
    print("Series shape:", series.shape)


if __name__ == "__main__":
    assert platform.system() == "Linux"
    assert os.path.isdir("series/")
    print(
        "The following files are available to print (by entering `series/<name>.csv`):"
    )
    os.system("ls series/")
    PATH = input("Enter the path to a saved curve as a csv: ")
    RAW_SERIES = pd.read_csv(PATH)
    SERIES = np.array(RAW_SERIES)
    plot_to_terminal(SERIES)
