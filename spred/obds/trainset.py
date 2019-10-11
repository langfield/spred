""" Function to construct orderbook array for training. """
import os
import sys
import time
import json
import argparse
import itertools
import functools
import multiprocessing as mp

from typing import List

import numpy as np
import pandas as pd

from arguments import df_args


def gen_df(hours: int, trunc: int, save_path: str, source_dir: str) -> None:
    """
    Reads in a range of hour orderbook json files and computes a 3-sigma confidence
    interval for the random variable Y representing the max of RVs Y_1 and Y_2, which
    are the best ask and best bid prices at the next time step.

    Parameters
    ----------
    hours : ``int``.
        Function reads in all hour orderbook files from 0 to ``hours`` via filename.
    trunc : ``int``.
        The number of nonzero levels to include on each side.
    """

    # Validate paths.
    dirname = os.path.dirname(save_path)
    assert os.path.isdir(dirname)
    assert os.path.isdir(source_dir)

    print("Constructing training dataset from %d hours of tick-level data." % hours)
    print("Saving to file: %s" % save_path)
    start = time.time()
    pool = mp.Pool()
    hrs = range(hours)
    get_book_vecs = functools.partial(process_books, trunc=trunc, source_dir=source_dir)

    vecs: List[np.array] = []

    for i, book_vecs in enumerate(pool.imap_unordered(get_book_vecs, hrs), 1):
        sys.stderr.write("\rdone {0:%}".format(i / hours))
        vecs.extend(book_vecs)
    print("")

    hours_array = np.stack(vecs)
    hours_df = pd.DataFrame(hours_array)
    hours_df.to_csv(path_or_buf=save_path, index=False)
    print("Finished in %fs" % (time.time() - start))


def process_books(hour: int, trunc: int, source_dir: str) -> List[np.ndarray]:
    """
    Reads the specified orderbook json file and outputs statistics on the
    bid and ask distribution. Plots the ask price difference distribution.
    The ``subbook`` is a list of lists where the inner list has two elements.
    The first element is the price, and the second is the volume.

    Parameters
    ----------
    hour : ``int``.
        The set of 3600 timesteps of orderbooks to read in, indexed from 0 in filename.
    trunc : ``int``.
        The number of nonzero levels to include on each side.

    Returns
    -------
    vecs : ``List[np.ndarray]``.
        The array of all orderbook feature vectors from this hour of tick data.
        Shape: ``(<hour_len>, 6)``.
    """

    assert os.path.isdir(source_dir)
    path = os.path.join(source_dir, "out_%d.json" % hour)
    with open(path) as json_file:
        raw_books = json.load(json_file)

    # Convert the keys (str) of ``raw_books`` to integers.
    books = {}
    for i, index_book_pair in enumerate(raw_books.items()):
        book_index_str, book = index_book_pair
        books.update({i: book})
        assert i == int(book_index_str)

    vecs: List[np.array] = []

    for i, book in books.items():
        if i > 0:
            bidbook = book["bids"]
            askbook = book["asks"]
            bidbook = bidbook[:trunc]
            askbook = askbook[:trunc]
            try:
                assert trunc <= min(len(bidbook), len(askbook))
            except AssertionError:
                print("args.trunc: %d" % trunc)
                print("len(bidbook): %d" % len(bidbook))
                print("len(askbook): %d" % len(askbook))
                assert trunc <= min(len(bidbook), len(askbook))
            prev_bidbook = books[i - 1]["bids"]
            prev_askbook = books[i - 1]["asks"]
            best_bid = bidbook[0][0]
            best_ask = askbook[0][0]
            prev_best_bid = prev_bidbook[0][0]
            prev_best_ask = prev_askbook[0][0]

            prev_bid_level_indices = []
            prev_ask_level_indices = []
            bid_level_indices = []
            ask_level_indices = []
            bid_sizes = []
            ask_sizes = []

            for bid_level, bid_size in bidbook:
                prev_bid_level_index = round(bid_level - prev_best_bid, 2)
                bid_level_index = round(bid_level - best_bid, 2)
                prev_bid_level_indices.append(prev_bid_level_index)
                bid_level_indices.append(bid_level_index)
                bid_sizes.append(bid_size)

            for ask_level, ask_size in askbook:
                prev_ask_level_index = round(ask_level - prev_best_ask, 2)
                ask_level_index = round(ask_level - best_ask, 2)
                prev_ask_level_indices.append(prev_ask_level_index)
                ask_level_indices.append(ask_level_index)
                ask_sizes.append(ask_size)

            vec = list(
                itertools.chain(
                    prev_bid_level_indices,
                    bid_level_indices,
                    bid_sizes,
                    prev_ask_level_indices,
                    ask_level_indices,
                    ask_sizes,
                )
            )

            feature_array = np.array(vec)
            vecs.append(feature_array)

    assert len(vecs) == len(books) - 1

    return vecs


def main() -> None:
    """ Construct training data in csv format. """
    parser = argparse.ArgumentParser()
    parser = df_args(parser)
    args = parser.parse_args()

    gen_df(
        hours=args.hours,
        trunc=args.trunc,
        save_path=args.save_path,
        source_dir=args.source_dir,
    )


if __name__ == "__main__":
    main()
