""" Kraken orderbook scraper. """
import os
import sys
import time
import json
import shutil
import argparse
import datetime
import tempfile
import functools
import multiprocessing as mp
from typing import Any, Dict, Tuple

from tor_request import TorRequest

# pylint: disable=bad-continuation, too-many-locals

DEBUG = False
PARSE_TIME = True


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


def until(date: datetime.datetime, pid: int = -1) -> None:
    """
    Wait until the requested start date. Print ``diff`` at the end to
    see margin of error.

    Parameters
    ----------
    date : ``datedate.datetime``.
        Wait until this utc time.
    """
    if DEBUG:
        print("PID: %d: Waiting until %s." % (pid, str(date)))
    while 1:
        if datetime.datetime.utcnow() > date:
            break
        diff = (date - datetime.datetime.utcnow()).total_seconds()
        wait = max(max(diff - 0.01, 0), 0.001)
        time.sleep(wait)
    diff = (date - datetime.datetime.utcnow()).total_seconds()


def parse(
    date_count: Tuple[int, int, int, datetime.datetime, int],
    interval: int,
    url: str,
    padding: int,
    save_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Schedules and runs parses at each time in dates, and stores the dictionary
    of resultant data in ``orderbook_dict``.

    Parameters
    ----------
    date_count : ``Tuple[int, int, int, datetime.datetime, int]``.
        Tuple of the process id, proxy port, ctrl port, unix time at which to begin
        parsing the given url, and the number of parses to execute.
    interval : ``int``.
        Interval between parses in seconds.
    url : ``str``.
        Page to scrape json from.
    padding : ``int``.
        How many seconds to wait for ``TorRequest()`` to start up.
    save_dir : ``str``.
        Directory in which to save the books.

    Returns
    -------
    books : ``Dict[int, Dict[str, Any]]``.
        Dictionary mapping dates to data.
    """

    pid, proxy_port, ctrl_port, date, num_requests = date_count
    bookid = 0
    data_dir = tempfile.mkdtemp()

    until(date - datetime.timedelta(seconds=padding), pid=pid)
    books: Dict[str, Dict[str, Any]] = {}

    print(
        "PID: %d: Done with first wait. If this hangs or takes more than 20s, " % pid
        + "make sure tor and SOCKS dependencies are installed. Try running duoscape."
    )

    with TorRequest(
        proxy_port=proxy_port, ctrl_port=ctrl_port, data_dir=data_dir
    ) as tor:

        if DEBUG:
            print("PID: %d: Opened TorRequest." % pid)
            sys.stdout.flush()

        try:
            while 1:
                # We split the ``until()`` call since ``TorRequest()`` takes around 4s.
                until(date, pid=pid)
                start = time.time()
                now = date

                # Set date to next interval.
                date += datetime.timedelta(seconds=2 * num_requests * interval)

                for _ in range(num_requests):
                    try:
                        response = tor.get(url)
                        content = response.text
                    except Exception as exc:
                        print(exc)
                        raise ValueError(str(exc))
                    end = datetime.datetime.utcnow()
                    error = end - now

                    data = json.loads(content)

                    if DEBUG:
                        print(data["origin"].split(",")[0])

                    rounded_stamp = now.strftime("%H:%M:%S")
                    stamp = now.strftime("%H:%M:%S:%f")
                    end_stamp = end.strftime("%H:%M:%S:%f")
                    books[rounded_stamp] = data
                    if PARSE_TIME:
                        print(
                            "PID: %d: Parse start: '%s' end: '%s' with margin: '%s'"
                            % (pid, stamp, end_stamp, error)
                        )
                        sys.stdout.flush()

                    now += datetime.timedelta(seconds=interval)
                    until(now, pid=pid)

                filename = "book_%d_pid_%d.json" % (bookid, pid)
                save_path = os.path.join(save_dir, filename)
                with open(save_path, "w") as book_file:
                    json.dump(books, book_file)
                books = {}
                bookid += 1

        except Exception as error:
            print(error)
            shutil.rmtree(data_dir)
            raise ValueError(str(error))


def main(args: argparse.Namespace) -> None:
    """ Continuously scrape the specified orderbook and save to a json file. """

    # Set the scrape interval delay, and the number of books to parse per file.
    delay = 1
    padding = 13
    num_parses = 30  # 600
    num_workers = 2
    starting_proxy_port = 9060
    starting_ctrl_port = 9061

    # Takes about 10s minimum to execute the ``TorRequests()`` call.
    assert padding > 11

    # Make sure the directory exists. Create it if not.
    if not os.path.isdir(args.dir):
        os.mkdir(args.dir)

    url = "https://api.cryptowat.ch/markets/kraken/ethusd/orderbook"
    url = "http://httpbin.org/ip"
    start = round_time(date=datetime.datetime.utcnow(), granularity=1)
    start += datetime.timedelta(seconds=padding + 2)

    # The first ``remainder`` workers each make ``iterations + 1`` parses, the rest
    # make ``iterations`` parses.
    proxy_ports = [starting_proxy_port + (i * 10) for i in range(num_workers)]
    ctrl_ports = [starting_ctrl_port + (i * 10) for i in range(num_workers)]
    dates = [
        start + datetime.timedelta(seconds=i * num_parses) for i in range(num_workers)
    ]
    counts = [num_parses for _ in range(num_workers)]
    pids = range(num_workers)
    assert len(counts) == len(dates) == num_workers

    date_counts = zip(pids, proxy_ports, ctrl_ports, dates, counts)
    sfn = functools.partial(
        parse, url=url, interval=delay, padding=padding, save_dir=args.dir
    )
    pool = mp.Pool(num_workers)
    pool.map(sfn, date_counts)


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
