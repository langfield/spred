""" Kraken orderbook scraper. """
import os
import time
import json
import argparse
from urllib.request import urlopen


def main(args: argparse.Namespace) -> None:
    """ Continuously scrape the specified orderbook and save to a json file. """

    # Set the scrape interval delay, and the number of timesteps per file.
    delay = 1.0
    sec_per_file = 3600

    # Make sure the directory exists. Create it if not.
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)

    url_str = "https://api.cryptowat.ch/markets/kraken/ethusd/orderbook"
    index = 0
    out = {}
    start = time.time()
    file_count = args.start

    while True:
        url = urlopen(url_str)
        content = url.read()
        data = json.loads(content)
        print("  Finished parsing index %d.\r" % index, end="")

        # Construct ``order_dict`` from the json input.
        seq_num = data["result"]["seqNum"]
        _allowance = data["allowance"]
        cur_time = time.time()
        asks = data["result"]["asks"]
        bids = data["result"]["bids"]
        order_dict = {"seq": seq_num, "time": cur_time, "asks": asks, "bids": bids}

        out.update({index: order_dict})
        index += 1

        # Write to file, and reset ``out`` dict.
        if index % sec_per_file == 0:
            path = os.path.join(args.dir, "out_%d.json" % file_count)
            with open(path, "w") as file_path:
                json.dump(out, file_path)
                print("\n  Dumped file %d." % file_count)
            file_count += 1
            index = 0
            out = {}

        time.sleep(delay - ((time.time() - start) % delay))

    print(out)


def get_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """ Parse the save directory for scraped orderbooks. """
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    return parser


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER = get_args(PARSER)
    ARGS = PARSER.parse_args()
    main(ARGS)
