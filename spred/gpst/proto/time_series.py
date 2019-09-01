import numpy as np
import pandas as pd


def sin(time: np.ndarray) -> np.ndarray:
    return np.sin(time)


def constant(time: np.ndarray) -> np.ndarray:
    return [0] * time.shape[0]


if __name__ == "__main__":

    x_min = 0
    x_max = 100
    num_steps = 10000
    dim = 40

    # x vals
    time = np.arange(x_min, x_max, (x_max - x_min) / num_steps)
    print("Number of data points:", time.shape[0])
    price = sin(time)  # Change function to modify time series.
    zeros = np.ones(num_steps)
    df = pd.DataFrame({"Price": price})
    df = df[[col for col in df.columns for i in range(dim)]]
    df.to_csv("sample_data.csv", index=False)
