""" Kraken orderbook scraper. """
import os
import sys
import time
import json
import argparse
import datetime
import functools
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from typing import List, Any, Dict, Tuple, Generator

from torrequest import TorRequest

# pylint: disable=bad-continuation


def round_time(date: datetime.datetime, granularity: int) -> datetime.datetime:
    """
    Round a datetime object to any time lapse in seconds.

    Parameters
    ----------
    date : ``datetime.datetime``.
        A timestamp.
    granularity : ``int``.
        Closest number of seconds to round to, default 1 minute.
    """

    seconds = (date.replace(tzinfo=None) - date.min).seconds
    rounding = (seconds + granularity / 2) // granularity * granularity
    rounded = date + datetime.timedelta(0, rounding - seconds, -date.microsecond)

    return rounded


def until(date: datetime.datetime) -> None:
    """
    Wait until the requested start date. Print ``diff`` at the end to
    see margin of error.

    Parameters
    ----------
    date : ``datedate.datetime``.
        Wait until this utc time.
    """
    while 1:
        if datetime.datetime.utcnow() > date:
            break
        diff = (date - datetime.datetime.utcnow()).total_seconds()
        # TODO: Fix.
        # Get within a hudredth of a second, then do milliseconds.
        wait = max(max(diff - 0.01, 0), 0.001)
        time.sleep(wait)
    diff = (date - datetime.datetime.utcnow()).total_seconds()


def schedule(
    date_count: Tuple[int, datetime.datetime, int],
    interval: datetime.timedelta,
    url: str,
    padding: int,
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
    padding : ``int``.
        How many seconds to wait for ``TorRequest()`` to start up.

    Returns
    -------
    books : ``Dict[int, Dict[str, Any]]``.
        Dictionary mapping dates to data.
    """

    pid, date, num_requests = date_count
    until(date - datetime.timedelta(seconds=padding))
    books: Dict[int, Dict[str, Any]] = {}

    with TorRequest() as tor:

        # We split the ``until()`` call since ``TorRequest()`` takes around 4s.
        until(date)
        start = time.time()

        # TODO: round ``now`` to nearest millisecond.
        now = date
        for _ in range(num_requests):
            try:
                response = tor.get(url)
                content = response.text
            except Exception as exc:
                print(exc)
                raise ValueError(str(exc))
            data = json.loads(content)
            books[now] = data
            stamp = now.strftime("%H:%M:%S")
            if pid == 0:
                print(
                    "PID: %d  \tParsed at time %s with true time elapsed %ds."
                    % (pid, stamp, time.time() - start)
                )
            sys.stdout.flush()
            now += interval
            until(now)

    return books


def runpool(
    start: datetime.datetime,
    url: str,
    delay: int,
    padding: int,
    num_parses: int,
    num_workers: int,
) -> List[Dict[int, Dict[str, Any]]]:
    """
    Scrape ``url`` every ``delay`` seconds ``num_parses`` times starting in
    ``padding`` seconds using ``num_workers`` different processes all running via
    distinct Tor connections, saving output to ``directory``.

    Parameters
    ----------
    start : ``datetime.datetime``.
        When to start parsing.
    url : ``str``.
        Page to scrape json from.
    delay : ``int``.
        Parse frequency.
    padding : ``int``.
        How many seconds to wait for ``TorRequest()`` to start up.
    num_parses : ``int``.
        Total across all processes.
    num_workers : ``int``.
        Processes in the pool.
    """

    print("Instantiating pool.")

    # Make sure ``start`` is sufficently far in the future.
    horizon = 2 * padding
    assert start - datetime.datetime.utcnow() > datetime.timedelta(seconds=horizon)

    # The first ``remainder`` workers each make ``iterations + 1`` parses, the rest
    # make ``iterations`` parses.
    iterations = num_parses // num_workers
    rem = num_parses % num_workers
    dates = [start + datetime.timedelta(seconds=i * delay) for i in range(num_workers)]
    counts = [iterations + 1 if i < rem else iterations for i in range(num_workers)]
    pids = range(num_workers)
    print("Sum of counts:", sum(counts))
    assert sum(counts) == num_parses
    assert len(counts) == len(dates) == num_workers

    date_counts = zip(pids, dates, counts)
    delta = datetime.timedelta(seconds=num_workers * delay)
    sfn = functools.partial(schedule, url=url, interval=delta, padding=padding)
    pool = mp.Pool(num_workers)
    bookdicts: List[Dict[int, Dict[str, Any]]] = pool.map(sfn, date_counts)

    return bookdicts


def main(args: argparse.Namespace) -> None:
    """ Continuously scrape the specified orderbook and save to a json file. """

    # Set the scrape interval delay, and the number of books to parse per file.
    url = "https://api.cryptowat.ch/markets/kraken/ethusd/orderbook"
    delay = 1
    padding = 6
    num_parses = 60
    num_workers = 10

    # Takes about 4s minimum to execute the ``TorRequests()`` call.
    assert padding > 5

    # Make sure the directory exists. Create it if not.
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)

    file_count = args.start

    def seeds(start: datetime.datetime, interval: int) -> Generator[int, None, None]:
        """ Generate hour timestamps. """
        now = start
        while 1:
            print("Seeding.")
            yield now
            now += datetime.timedelta(hours=interval)

    # DEBUG
    start = round_time(date=datetime.datetime.utcnow(), granularity=1)
    start += datetime.timedelta(seconds=3 * padding)

    pool_fn = functools.partial(
        runpool,
        url=url,
        delay=delay,
        padding=padding,
        num_parses=num_parses,
        num_workers=num_workers,
    )

    metapool = ThreadPool(2)
    print("Imapping.")
    book = list(metapool.imap(pool_fn, seeds(start, 1)))


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
