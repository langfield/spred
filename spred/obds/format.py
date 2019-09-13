""" Kraken orderbook plot utility. """
import json
from typing import List

import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position, ungrouped-imports
import seaborn as sns
import matplotlib.pyplot as plt


def collect_gaps(subbook: List[List[float]]) -> List[float]:
    """
    Computes the list of consecutive differences for price levels in a subbook.
    Removes gaps greater than 100 for plotting purposes.

    Parameters
    ----------
    subbook : ``List[List[float]], required.
        Either askbook or bidbook, each element is a two-element list
        where the first item is the price level, and the second item is
        the volume.

    Returns
    -------
    gaps : ``List[float]``.
        The consecutive price differences in the list, computed from level 0
        outward (from best price to worst).
    """
    gaps: List[float] = []
    levels = [pair[0] for pair in subbook]
    for j, level in enumerate(levels):
        if j > 0:
            gap = level - levels[j - 1]
            gap = round(gap, 2)
            if gap <= 100:
                gaps.append(gap)
    return gaps


def main() -> None:
    """
    Reads the specified orderbook json file and outputs statistics on the
    bid and ask distribution. Plots the ask price difference distribution.
    """

    with open("results/out_0.json") as json_file:
        raw_books = json.load(json_file)
        print("Loaded json.")

    # Convert the keys (str) of ``raw_books`` to integers.
    books = {}
    for i, index_book_pair in tqdm(enumerate(raw_books.items())):
        book_index_str, book = index_book_pair
        books.update({i: book})
        assert i == int(book_index_str)

    bid_lens: List[int] = []
    ask_lens: List[int] = []
    ask_gaps: List[float] = []
    bid_gaps: List[float] = []
    best_ask_deltas: List[float] = []
    best_bid_deltas: List[float] = []

    # Loop over timesteps.
    for i, book in tqdm(books.items()):

        # Loop over orderbook key-value pairs.
        for side, subbook in book.items():

            if side == "asks":
                askbook = subbook
                ask_lens.append(len(askbook))
                ask_gaps.extend(collect_gaps(askbook))

                if i > 0:
                    prev_askbook = books[i - 1]["asks"]
                    best_ask = askbook[0][0]
                    prev_best_ask = prev_askbook[0][0]
                    best_ask_delta = best_ask - prev_best_ask
                    best_ask_delta = round(best_ask_delta, 2)
                    best_ask_deltas.append(best_ask_delta)

            elif side == "bids":
                bidbook = subbook
                bid_lens.append(len(bidbook))
                bid_gaps.extend(collect_gaps(bidbook))

                if i > 0:
                    prev_bidbook = books[i - 1]["bids"]
                    best_bid = bidbook[0][0]
                    prev_best_bid = prev_bidbook[0][0]
                    best_bid_delta = best_bid - prev_best_bid
                    best_bid_delta = round(best_bid_delta, 2)
                    best_bid_deltas.append(best_bid_delta)

    # Matplotlib setup.
    _, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)

    hex_cube = sns.color_palette("cubehelix", 8).as_hex()

    # Generate histogram of the gaps between ask prices in orderbook.
    num_bins = len(set(ask_gaps))
    sb_ax = sns.distplot(ask_gaps, bins=num_bins, kde=False, ax=axes[0, 0], color=hex_cube[5])
    sb_ax.set_yscale("log")

    # Generate histogram of the gaps between bid prices in orderbook.
    num_bins = len(set(bid_gaps))
    sb_ax = sns.distplot(bid_gaps, bins=num_bins, kde=False, ax=axes[0, 0], color=hex_cube[6], label="Bid/Ask Gap Distribution")
    sb_ax.set_yscale("log")

    # Generate histogram of the gaps between bid prices in orderbook.
    num_bins = len(set(best_ask_deltas))
    print("Num bins:", num_bins)
    print("Best ask deltas:", best_ask_deltas)
    sb_ax = sns.distplot(best_ask_deltas, bins=None, kde=False, ax=axes[0, 1], color=hex_cube[4], label="Best Ask Delta Distribution")
    sb_ax.set_yscale("log")

    plt.savefig("bid_ask_price_gap_dist.svg")

    print("Min of bids:", min(bid_lens))
    print("Max of bids:", max(bid_lens))
    print("Mean of bids:", np.mean(bid_lens))
    print("Standard deviation of bids:", np.std(bid_lens))
    print("")
    print("Min of asks:", min(ask_lens))
    print("Max of asks:", max(ask_lens))
    print("Mean of asks:", np.mean(ask_lens))
    print("Standard deviation of asks:", np.std(ask_lens))


if __name__ == "__main__":
    main()
