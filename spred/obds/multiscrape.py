""" Kraken orderbook scraper. """
import os
import sys
import time
import json
import argparse
import datetime
import functools
import multiprocessing as mp
from typing import List, Any, Dict, Tuple
from urllib.request import urlopen, Request

from torrequest import TorRequest

# pylint: disable=bad-continuation


def round_time(dt: datetime.datetime, granularity: int) -> datetime.datetime:
    """
    Round a datetime object to any time lapse in seconds.

    Parameters
    ----------
    dt : ``datetime.datetime``.
        A timestamp.
    granularity : ``int``.
        Closest number of seconds to round to, default 1 minute.
    """

    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + granularity / 2) // granularity * granularity
    rounded = dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)

    return rounded


def schedule(
    date_count: Tuple[int, datetime.datetime, int],
    interval: datetime.timedelta,
    url: str,
) -> Dict[int, Dict[str, any]]:
    """
    Schedules and runs parses at each time in dates, and stores the dictionary
    of resultant data in ``orderbook_dict``.

    Parameters
    ----------
    date_count : ``Tuple[int, int]``.
        Tuple of the unix time at which to begin parsing the given url, and the number
        of parses to execute.
    interval : ``int``.
        Interval between parses in seconds.
    url : ``str``.
        Page to scrape json from.

    Returns
    -------
    books : ``Dict[int, Dict[str, Any]]``.
        Dictionary mapping dates to data.
    """

    pid, date, n = date_count
    req = Request(url, data=None, headers={"User-Agent": "IgnoreMe."})

    with TorRequest() as tr:

        # Wait until the requested start date.
        while 1:
            if datetime.datetime.utcnow() > date:
                break
            else:
                diff = (date - datetime.datetime.utcnow()).total_seconds()
                wait = max(diff, 0.5)
            print("Waiting for %fs." % wait)
            time.sleep(wait)

        books: Dict[int, Dict[str, Any]] = {}

        # TODO: round ``now`` to nearest millisecond.
        now = date
        for i in range(n):
            try:
                page = tr.get(url)
                content = page.read()
            except Exception as e:
                print(e.headers)
                raise ValueError(str(e))
            data = json.loads(content)
            books[now] = data
            stamp = now.strftime("%H:%M:%S")
            if pid == 0:
                print("PID: %d  \tParsed at time %s." % (pid, stamp))
            sys.stdout.flush()
            now += interval

    return books


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
    start = round_time(dt=datetime.datetime.utcnow(), granularity=1)
    start += datetime.timedelta(seconds=5)
    file_count = args.start
    num_workers = 2

    # The first ``remainder`` workers each make ``iterations + 1`` parses, the rest
    # make ``iterations`` parses.
    iterations = sec_per_file // num_workers
    rem = sec_per_file % num_workers
    dates = [start + datetime.timedelta(seconds=i) for i in range(num_workers)]
    counts = [iterations + 1 if i < rem else iterations for i in range(num_workers)]
    pids = [i for i in range(num_workers)]
    print("Sum of counts:", sum(counts))
    assert sum(counts) == sec_per_file
    assert len(counts) == len(dates) == num_workers

    date_counts = zip(pids, dates, counts)
    sfn = functools.partial(schedule, url=url, interval=datetime.timedelta(seconds=1))
    pool = mp.Pool(num_workers)
    bookdicts: List[Dict[int, Dict[str, Any]]] = pool.map(sfn, date_counts)

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
