import numpy as np
import pandas as pd

x_min = 0
x_max = 100
num_steps = 10000
# x vals
time = np.arange(x_min, x_max, (x_max - x_min) / num_steps)
print('Number of data points:', time.shape[0])

price = np.sin(time)
price = [0] * num_steps

zeros = np.ones(num_steps)
df = pd.DataFrame({'Price': price})
df = df[[col for col in df.columns for i in range(10)]]
df.to_csv('sample_data.csv', index=False)
