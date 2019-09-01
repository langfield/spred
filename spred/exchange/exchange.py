import pandas as pd
from datetime import datetime

class Exchange:
    def __init__(self, file, initial_balance=1000):
        self.trade_data = pd.read_csv(file)
        self.funds = initial_balance
        self.time = datetime.strptime(self.trade_data.iloc[0][0], '%Y-%m-%d %H:%M:%S.%f')
        self.coins = 0

    def set_time(self, time):
        stime = time.strftime('%Y-%m-%d %H:%M:%S.%f')
        if self.trade_data['Date'].isin([stime]).any():
            self.time = time
        else:
            print('Time out of bounds')

    def print_time(self):
        print(self.time)

    def get_price(self):
        stime = self.time.strftime('%Y-%m-%d %H:%M:%S.%f')
        return self.trade_data.loc[
            self.trade_data['Date'] == stime]['Close'].values[0]

    # amount in fiat
    def buyf(self, amount):
        if amount > self.funds or amount <= 0:
            return False

        self.funds -= amount
        self.coins += amount / self.get_price()

        return True

    # amount in coins
    def buyc(self, amount):
        price = self.get_price()
        if amount * price > self.funds or amount <= 0:
            return False

        self.coins += amount
        self.funds -= amount * price

        return True

    # amount in coins
    def sellc(self, amount):
        if amount > self.coins or amount <= 0:
            return False

        self.coins -= amount
        self.funds += amount * self.get_price()

        return True

    # amount in fiat
    def sellf(self, amount):
        price = self.get_price()
        if amount / price > self.coins or amount <= 0:
            return False

        self.coins -= amount / price
        self.funds += amount

        return True

    def print_account(self):
        print('-------Account details-------')
        print('Time : ' + str(self.time))
        print('Price: ' + str(self.get_price()))
        print('Funds: ' + str(self.funds))
        print('Coins: ' + str(self.coins))
        print('-----------------------------')


e = Exchange('ETHUSDT.csv', 1000)

e.set_time(datetime(2017, 10, 4, 14, 55))
e.print_account()

e.buyf(100)
e.print_account()

e.set_time(datetime(2017, 12, 26))
e.sellc(.3)
e.print_account()