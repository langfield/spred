""" Kraken orderbook plot utility. """
import json
import collections
from typing import List, Set, Dict

import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position, ungrouped-imports
import seaborn as sns
import matplotlib.pyplot as plt


def collect_gaps(subbook: List[List[float]], depth: int = -1) -> List[float]:
    """
    Computes the list of consecutive differences for price levels in a subbook.
    Removes gaps greater than 100 for plotting purposes.

    Parameters
    ----------
    subbook : ``List[List[float]], required.
        Either askbook or bidbook, each element is a two-element list
        where the first item is the price level, and the second item is
        the volume.
    depth : ``int``, optional.
        How far in the subbook from best price to go. Passing ``-1`` takes all gaps.

    Returns
    -------
    gaps : ``List[float]``.
        The consecutive price differences in the list, computed from level 0
        outward (from best price to worst).
    """
    gaps: List[float] = []
    levels = [pair[0] for pair in subbook]
    if depth >= 0:
        levels = levels[: min(depth, len(levels))]
    for j, level in enumerate(levels):
        if j > 0:
            gap = level - levels[j - 1]
            gap = round(gap, 2)
            if gap <= 100:
                gaps.append(gap)
    return gaps


def print_subbook_stats(
    gap_list: Dict[str, List[List[float]]],
    best_deltas: Dict[str, List[float]],
    best_in_prev: Dict[str, List[bool]],
    book_lens: Dict[str, List[int]],
) -> None:
    """
    Computes and prints subbook statistics.

    Parameters
    ----------
    gap_list : ``Dict[str, List[List[float]]]``.
        The keys are "bids" or "asks", and the values are a list whose
        length is the number of timesteps. Each item of a value is a list
        of all gaps starting at level 0 for that side.
    best_deltas : ``Dict[str, List[float]]``.
        List of changes in best price from time ``t`` to time ``t + 1``.
    best_in_prev : ``Dict[str, List[bool]]``.
        Maps subbook side to list of booleans telling whether or not the current
        best price level existed in the previous orderbook.
    book_lens : ``Dict[str, List[bool]]``.
        Maps subbook sides to their lengths.
    """
    for side, best_side_deltas in best_deltas.items():

        num_zero_deltas = 0
        num_pos_deltas = 0
        num_neg_deltas = 0
        for delta in best_side_deltas:
            if delta == 0:
                num_zero_deltas += 1
            elif delta > 0:
                num_pos_deltas += 1
            elif delta < 0:
                num_neg_deltas += 1

        best_delta_mean = np.mean(best_side_deltas)
        best_delta_stddev = np.std(best_side_deltas)
        two_sigma_lower = best_delta_mean - (2 * best_delta_stddev)
        two_sigma_upper = best_delta_mean + (2 * best_delta_stddev)
        three_sigma_lower = best_delta_mean - (3 * best_delta_stddev)
        three_sigma_upper = best_delta_mean + (3 * best_delta_stddev)
        three_sigma_width = max(abs(three_sigma_lower), abs(three_sigma_upper))
        three_sigma_k = round(100 * three_sigma_width)

        # The i-th list in ``level_dists`` is all gaps at level i of this subbook.
        analysis_depth = 10
        level_dists: List[List[float]] = [[] for i in range(analysis_depth)]
        subbook_gap_lists = gap_list[side]

        # Iterate over timesteps. Note ``gap`` is a float.
        for subbook_gap_list in subbook_gap_lists:
            for i, gap in enumerate(subbook_gap_list[:analysis_depth]):
                level_dists[i].append(gap)

        # Contains the mean and stddev of each set of gaps at level i of this subbook.
        level_dist_stats = []

        # Iterate over levels in a subbook.
        for level_dist in level_dists:
            stats = {}
            stats["mean"] = np.mean(level_dist)
            stats["std"] = np.std(level_dist)
            level_dist_stats.append(stats)

        num_nonzero_deltas = num_pos_deltas + num_neg_deltas
        zero_delta_prop = num_zero_deltas / len(best_side_deltas)
        pos_delta_prop = num_pos_deltas / len(best_side_deltas)
        neg_delta_prop = num_neg_deltas / len(best_side_deltas)

        num_new_bests = len([status for status in best_in_prev[side] if not status])
        new_best_percent = 100 * num_new_bests / len(best_in_prev[side])
        new_best_change_percent = 100 * num_new_bests / num_nonzero_deltas

        print("\n%s statistics" % side[:-1])
        print("--------------")
        print("%% of best %s at prev zero-vol lvl: %f%%" % (side, new_best_percent))
        print("%% of best %s at prev zero-vol lvl " % side, end="")
        print("given change: %f%%\n" % new_best_change_percent)

        print("Zero best %s delta proportion: %f" % (side[:-1], zero_delta_prop))
        print("Positive best %s delta proportion: %f" % (side[:-1], pos_delta_prop))
        print("Negative best %s delta proportion: %f" % (side[:-1], neg_delta_prop))
        print("Positive best %s deltas: %d" % (side[:-1], num_pos_deltas))
        print("Negative best %s deltas: %d" % (side[:-1], num_neg_deltas))
        print("Total best %s deltas: %d\n" % (side[:-1], len(best_side_deltas)))

        print("Best delta mean: %f" % best_delta_mean)
        print("Best delta stddev: %f" % best_delta_stddev)
        print("Two sigma confidence interval: ", end="")
        print("[%f, %f]" % (two_sigma_lower, two_sigma_upper))
        print("Three sigma confidence interval: ", end="")
        print("[%f, %f]" % (three_sigma_lower, three_sigma_upper))
        print("Three sigma k-value: %d\n" % three_sigma_k) 

        print("Min of %s: %d" % (side, min(book_lens[side])))
        print("Max of %s: %d" % (side, max(book_lens[side])))
        print("Mean of %s: %f" % (side, np.mean(book_lens[side])))
        print("Standard deviation of %s: %f\n" % (side, np.std(book_lens[side])))

        print("Gap statistics for each level in subbook:")
        for i, level_stats in enumerate(level_dist_stats):
            mean = level_stats["mean"]
            std = level_stats["std"]
            print("\tlevel %d:\t mean: %f    \tstandard deviation: %f" % (i, mean, std))

        # DEBUG
        # HARDCODE
        for i, level_dist in enumerate(level_dists[:3]):
            print("\nLevel %d distribution:" % i)
            gap_freqs = collections.Counter(level_dist)
            gap_freq_items = sorted(gap_freqs.items(), key=lambda x: x[1], reverse=True)
            for gap, freq in gap_freq_items[:5]:
                print("Gap: %f \t Freq: %d" % (gap, freq))
        print("")


def generate_plots(
    gaps: Dict[str, List[float]], best_deltas: Dict[str, List[float]]
) -> None:
    """
    Generates seaborn plots for the given variables, which are hardcoded for now.

    Parameters
    ----------
    gaps : ``Dict[str, List[float]]``, required.
        The gap sizes between consecutive prices in all subbooks in all orderbooks.
    best_deltas : ``Dict[str, List[float]]``, required.
        The deltas (scalar differences) between best prices in consecutive orderbooks.
    """
    ask_gaps: List[float] = gaps["asks"]
    bid_gaps: List[float] = gaps["bids"]
    best_ask_deltas: List[float] = best_deltas["asks"]
    best_bid_deltas: List[float] = best_deltas["bids"]

    # Get color palette.
    hex_cube = sns.color_palette("cubehelix", 8).as_hex()

    # Generate histogram of the gaps between bid/ask prices in orderbook.
    _, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    num_bins = len(set(ask_gaps))
    sb_ax = sns.distplot(ask_gaps, bins=num_bins, kde=False, ax=axes, color=hex_cube[5])
    sb_ax.set_yscale("log")
    num_bins = len(set(bid_gaps))
    sb_ax = sns.distplot(bid_gaps, bins=num_bins, kde=False, ax=axes, color=hex_cube[6])
    sb_ax.set_title("Bid/Ask Gap Distribution")
    sb_ax.set_yscale("log")
    plt.savefig("bid_ask_price_gap_dist.svg")
    plt.clf()

    # Generate histogram of the deltas between best ask prices in consecutive orderbooks.
    _, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    num_bins = len(set(best_ask_deltas))
    sb_ax = sns.distplot(best_ask_deltas, kde=False, ax=axes, color=hex_cube[4])
    sb_ax.set_title("Best Ask Delta Distribution")
    sb_ax.set_yscale("log")
    plt.savefig("best_ask_deltas.svg")
    plt.clf()

    # Generate histogram of the deltas between best bid prices in consecutive orderbooks.
    _, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)
    num_bins = len(set(best_bid_deltas))
    sb_ax = sns.distplot(best_bid_deltas, kde=False, ax=axes, color=hex_cube[4])
    sb_ax.set_title("Best Bid Delta Distribution")
    sb_ax.set_yscale("log")
    plt.savefig("best_bid_deltas.svg")


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

    book_lens: Dict[str, List[int]] = {"bids": [], "asks": []}
    gaps: Dict[str, List[float]] = {"bids": [], "asks": []}
    best_deltas: Dict[str, List[float]] = {"bids": [], "asks": []}
    best_in_prev: Dict[str, List[bool]] = {"bids": [], "asks": []}
    gap_list: Dict[str, List[List[float]]] = {"bids": [], "asks": []}

    # Loop over timesteps.
    for i, book in tqdm(books.items()):

        # Loop over orderbook key-value pairs.
        for side, subbook in book.items():

            if side in ("asks", "bids"):
                book_lens[side].append(len(subbook))
                gaps[side].extend(collect_gaps(subbook))
                gap_list[side].append(collect_gaps(subbook))

                if i > 0:
                    prev_subbook: List[List[float]] = books[i - 1][side]
                    prev_lvls: Set[float] = {order[0] for order in prev_subbook}
                    best = subbook[0][0]
                    prev_best = prev_subbook[0][0]
                    best_delta = best - prev_best
                    best_delta = round(best_delta, 2)
                    best_deltas[side].append(best_delta)
                    best_in_prev[side].append(best in prev_lvls)

    generate_plots(gaps, best_deltas)
    print_subbook_stats(gap_list, best_deltas, best_in_prev, book_lens)


if __name__ == "__main__":
    main()
