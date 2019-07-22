import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot

stock_data = pd.read_csv('exchange/ETHUSDT_small.csv')
stock_data = stock_data.drop(columns=['Date'])
print(stock_data.head())

sns.pairplot(stock_data)
