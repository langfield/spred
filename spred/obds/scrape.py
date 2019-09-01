import urllib.request as request
import json
import time

delay = 1.0
sec_per_file = 3600

url_str = "https://api.cryptowat.ch/markets/kraken/ethusd/orderbook"
index = 0
out = {}
start = time.time()
file_count = 0

while True:
    url = request.urlopen(url_str)
    content = url.read()
    data = json.loads(content)

    seqNum = data["result"]["seqNum"]
    allowance = data["allowance"]
    cur_time = time.time()
    asks = data["result"]["asks"]
    bids = data["result"]["bids"]

    d = {
        "seq": seqNum,
        "time": cur_time,
        "asks": asks,
        "bids": bids
    }

    out.update({index:d})
    index += 1

    if index % sec_per_file == 0:
        with open("results/out_{}.json".format(file_count), 'w') as fp:
            json.dump(out, fp)
        file_count += 1
        index = 0
        out = {}

    time.sleep(delay - ((time.time() - start) % delay))

print(out)