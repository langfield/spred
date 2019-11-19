""" Kraken orderbook scraper. """
import os
import sys
import time
import json
import sched
import argparse
import datetime
import functools
import multiprocessing as mp
from typing import List, Any, Dict
from urllib.request import urlopen


def parse(token: int, url: str) -> Dict[str, Any]:
    """ Grab json from the given url. """
    page = urlopen(url)
    content = page.read()
    data = json.loads(content)
    stamp = datetime.datetime.utcfromtimestamp(token).strftime("%H:%M:%S")
    print("Parsed at time %s." % stamp)
    sys.stdout.flush()
    return token, data


def schedule(tokens: List[int], url: str) -> Dict[int, Dict[str, any]]:
    """
    Schedules and runs parses at each time in tokens, and stores the dictionary
    of resultant data in ``orderbook_dict``.

    Parameters
    ----------
    tokens : ``List[int]``.
        Integer unix times at which to parse the given url.
    url : ``str``.
        Page to scrape json from.

    Returns
    -------
    orderbook_dict : ``Dict[int, Dict[str, Any]]``.
        Dictionary mapping tokens to data.
    """
    s = sched.scheduler()
    books = [s.enterabs(token, 0, parse, (token, url)) for token in tokens]
    s.run()
    bookdict = dict(books)
    print(bookdict)
    return bookdict


def main(args: argparse.Namespace) -> None:
    """ Continuously scrape the specified orderbook and save to a json file. """

    # Set the scrape interval delay, and the number of timesteps per file.
    delay = 1.0
    sec_per_file = 3600

    # Make sure the directory exists. Create it if not.
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)

    url = "https://api.cryptowat.ch/markets/kraken/ethusd/orderbook"
    index = 0
    out = {}
    start = round(time.time()) + 10
    file_count = args.start

    tokens = [start + i for i in range(3600)]

    num_workers = 60
    tokenlist_map: Dict[int, List[int]] = {}
    for i, token in enumerate(tokens):
        worker_id = i % num_workers
        if worker_id not in tokenlist_map:
            tokenlist_map[worker_id] = [token]
        else:
            tokenlist_map[worker_id].append(token)
    assert len(tokenlist_map) == num_workers
    tokenlists: List[List[int]] = tokenlist_map.values()
    schedul = functools.partial(schedule, url=url)
    pool = mp.Pool(num_workers)
    bookdicts: List[Dict[int, Dict[str, Any]]] = pool.map(schedul, tokenlists)

    sys.exit()

    while True:
        page = urlopen(url)
        content = page.read()
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
