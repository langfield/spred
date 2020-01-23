import krakenex
from pykrakenapi import KrakenAPI


def get_api(api_key_path):
    with open(api_key_path, "r") as api_key_file:
        lines = api_key_file.readlines()
        api = krakenex.API(key=lines[0], secret=lines[1])
    return KrakenAPI(api)


if __name__ == "__main__":
    k = get_api("../API_keys.txt")
    ohlc, last = k.get_ohlc_data("BCHUSD")
    print(ohlc.shape)
