import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential # type: ignore
from keras.layers import Dense, LSTM

stock_data = pd.read_csv("samplespred/ETHUSDT_small.csv")
# stock_data = stock_data.drop(columns=['Date'])
# stock_data.info()
# sns.pairplot(stock_data)

stock_data["Average"] = (stock_data["High"] + stock_data["Low"]) / 2
stock_data["Volume"] = stock_data["Volume"] + 0.000001  # Avoid NaNs
stock_data["Average_ld"] = np.log(stock_data["Average"]) - np.log(
    stock_data["Average"]
).shift(1)
stock_data["Volume_ld"] = np.log(stock_data["Volume"]) - np.log(
    stock_data["Volume"]
).shift(1)
# stock_data.head(2)
stock_data = stock_data[1:]
print(stock_data.head(2))

input_feature = stock_data.iloc[:, [7, 8]].values
print(stock_data.iloc[:, [7, 8]].head(2))
input_feature
input_data = input_feature

plt.plot(input_feature[:, 0])
plt.title("Volume of stock sold")
plt.xlabel("Time")
plt.ylabel("Volume")
plt.show()

sc = MinMaxScaler(feature_range=(0, 1))
input_data[:, 0:2] = sc.fit_transform(input_feature[:, :])

lookback = 50
test_size = int(0.3 * len(stock_data))
xlist = []
ylist = []

for i in range(len(stock_data) - lookback - 1):
    t = []
    for j in range(0, lookback):
        t.append(input_data[[(i + j)], :])
    xlist.append(t)
    ylist.append(input_data[i + lookback, 1])

x = np.array(xlist)
y = np.array(ylist)
x_test = x[: test_size + lookback]

x = x.reshape(x.shape[0], lookback, 2)
x_test = x_test.reshape(x_test.shape[0], lookback, 2)
print(x.shape)
print(x_test.shape)

model = Sequential()
model.add(LSTM(units=30, return_sequences=True, input_shape=(x.shape[1], 2)))
model.add(LSTM(units=30, return_sequences=True))
model.add(LSTM(units=30))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x, y, epochs=50, batch_size=32)

predicted_value = model.predict(x_test)

plt.plot(input_data[lookback : test_size + (2 * lookback), 1][:50], color="green")
plt.plot(predicted_value[:50], color="red")
plt.title("Opening price of stocks sold")
plt.xlabel("Time (latest-> oldest)")
plt.ylabel("Stock Opening Price")
plt.show()
