import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plot = False

width = 100
num_steps = 10000
# x vals
time = np.arange(0, width, 100 / num_steps)
print('Number of data points:', time.shape[0])
# y vals
price = [0.5] * num_steps

if plot:
    plt.plot(time, price)
    plt.title('Sample Time Series')
    plt.xlabel('Time (min)')
    plt.ylabel('Price')
    plt.show()

zeros = np.ones(num_steps)
df = pd.DataFrame({'Price': price})
df = df[[col for col in df.columns for i in range(60)]]
df.to_csv('sin.csv', index=False)
