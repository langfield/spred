""" Kraken orderbook plot utility. """
import json
import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position, ungrouped-imports
import seaborn as sb
import matplotlib.pyplot as plt


def main() -> None:
    """
    Reads the specified orderbook json file and outputs statistics on the
    bid and ask distribution. Plots the ask price difference distribution.
    """

    with open("results/out_0.json") as json_file:
        data = json.load(json_file)
        print("Loaded json.")

    bid_lens = []
    ask_lens = []
    ask_diffs = []

    # Loop over timesteps.
    # pylint: disable=too-many-nested-blocks
    for _, key_value in tqdm(enumerate(data.items())):
        _, value = key_value

        # Loop over orderbook key-value pairs.
        for subkey, subvalue in value.items():

            if subkey == "asks":
                ask_lens.append(len(subvalue))
                asks = [pair[0] for pair in subvalue]
                for j, ask in enumerate(asks):
                    if j > 0:
                        diff = ask - asks[j - 1]
                        diff = round(diff, 2)
                        # Remove extreme values for plotting purposes.
                        if diff > 100:
                            # print("prev:", asks[j - 1])
                            # print("ask:", ask)
                            pass
                        else:
                            ask_diffs.append(diff)

            elif subkey == "bids":
                bid_lens.append(len(subvalue))

    # Generate histogram of the diffs between ask prices in orderbook.
    num_bins = len(set(ask_diffs))
    sb_ax = sb.distplot(ask_diffs, bins=num_bins, kde=False)
    sb_ax.set_yscale("log")
    plt.savefig("ask_price_diff_dist.svg")

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
