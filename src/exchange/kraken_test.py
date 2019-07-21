import krakenex
from pykrakenapi import KrakenAPI

api = krakenex.API()
api.load_key('key.txt')
k = KrakenAPI(api)

bal = k.get_account_balance()
print(bal)
