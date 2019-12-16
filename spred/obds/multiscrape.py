""" Kraken orderbook scraper. """
import os
import sys
import time
import json
import argparse
import datetime
import functools
import multiprocessing as mp
from time import mktime
from queue import Full as QueueFull
from queue import Empty as QueueEmpty
from typing import List, Any, Dict, Tuple, Generator
from contextlib import closing

from torrequest import TorRequest

from lazy import pool_imap_unordered

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
    inq: mp.Queue, outq: mp.Queue, url: str, padding: int
) -> Dict[int, Dict[str, any]]:
    """
    Schedules and runs parses at each time in dates, and stores the dictionary
    of resultant data in ``orderbook_dict``.

    Parameters
    ----------
    date_count : ``Tuple[int, datetime.datetime, int]``.
        Tuple of the process ID, the unix time at which to begin parsing the given
        url, and the number of parses to execute.
    url : ``str``.
        Page to scrape json from.
    padding : ``int``.
        How many seconds to wait for ``TorRequest()`` to start up.

    Returns
    -------
    book : ``Dict[int, Dict[str, Any]]``.
        Dictionary mapping UNIX epochs to data.
    """

    while 1:
        date_count = inq.get()
        pid, date, num_requests = date_count
        until(date - datetime.timedelta(seconds=padding))
        book: Dict[datetime.datetime, Dict[str, Any]] = {}

        with closing(TorRequest()) as tor:

            # We split the ``until()`` call since ``TorRequest()`` takes around 4s.
            until(date)
            start = time.time()

            # TODO: round ``now`` to nearest millisecond.
            now = date
            try:
                response = tor.get(url)
                content = response.text
                print(content)
            except Exception as exc:
                print(exc)
                raise ValueError(str(exc))

        data = json.loads(content)
        unix_secs = mktime(now.timetuple())
        book[unix_secs] = data
        stamp = now.strftime("%H:%M:%S")

        print(
            "PID: %d  \tParsed at time %s with true time elapsed %ds."
            % (pid, stamp, time.time() - start)
        )
        sys.stdout.flush()
        outq.put(book)


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

    Returns
    -------
    all_books : ``Dict[datetime.datetime, Dict[str, Any]]``.
        Joined dictionary of all orderbooks for the given duration, with dates as keys.
    """

    url = "https://api.cryptowat.ch/markets/kraken/ethusd/orderbook"
    url = "http://httpbin.org/ip"
    start = round_time(date=datetime.datetime.utcnow(), granularity=1)
    start += datetime.timedelta(seconds=2 * padding)
    file_count = args.start
    stamp = start.strftime("%H:%M:%S")
    print("Instantiating pool to parse at %s." % stamp)
    sys.stdout.flush()

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
    sfn = functools.partial(schedule, url=url, padding=padding)
    pool = mp.Pool(num_workers)

    # Get books and merge dicts.
    # dicts: List[Dict[datetime.datetime, Dict[str, Any]]] = pool.map(sfn, date_counts)
    dicts = pool_imap_unordered(sfn, date_counts, num_workers)
    all_books: Dict[datetime.datetime, Dict[str, Any]] = {}
    for d in dicts:
        all_books.update(d)
    print("Returning all books.")
    sys.stdout.flush()

    return all_books


def seeds(
    start: datetime.datetime, delta: datetime.timedelta
) -> Generator[datetime.datetime, None, None]:
    """ Generate timestamps. """
    now = start
    while 1:
        yield now
        now += delta


def main(args: argparse.Namespace) -> None:
    """ Continuously scrape the specified orderbook and save to a json file. """

    # Set the scrape interval delay, and the number of books to parse per file.
    url = "https://api.cryptowat.ch/markets/kraken/ethusd/orderbook"
    delay = 1
    padding = 6
    num_parses = 20
    num_workers = 20
    num_metaprocesses = 2

    # Takes about 4s minimum to execute the ``TorRequests()`` call.
    assert padding > 5

    # Make sure the directory exists. Create it if not.
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)

    file_count = args.start

    # Start ``3 * padding`` seconds from now.
    start = round_time(date=datetime.datetime.utcnow(), granularity=1)
    start += datetime.timedelta(seconds=3 * padding)

    delta = datetime.timedelta(seconds=num_parses * delay)

    dateseeds = seeds(start, delta)
    itr = iter(dateseeds)

    # Create the list of processes.
    sfn = functools.partial(schedule, url=url, interval=delta, padding=padding)
    inqs = [mp.Queue(2) for _ in range(num_workers)]
    outqs = [mp.Queue() for _ in range(num_workers)]
    processes = [
        mp.Process(target=sfn, args=(inq, outq)) for inq, outq in zip(inqs, outqs)
    ]
    for process in processes:
        process.start()


    try:
        seed = next(itr)
        print("Seed:", seed)
        while True:

            # Find a way to pass in a continuous stream of seeds to all
            # the inqs. We need this loop to run indefinitely.
            for inq, outq in zip(inqs, outqs):
                try:
                    inq.put(seed)
                    seed = next(itr)
                except QueueFull:
                    while True:
                        try:
                            result = outq.get(False)
                            yield result
                        except QueueEmpty:
                            break
    except StopIteration:
        pass

    """
    while 1:
        out = next(twinparse)
        path = os.path.join(args.dir, "out_%d.json" % file_count)
        with open(path, "w") as file_path:
            json.dump(out, file_path)
            print("\n  Dumped file %d." % file_count)
        file_count += 1
    """


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
