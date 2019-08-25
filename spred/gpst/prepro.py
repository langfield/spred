""" Preprocess a raw time-series dataset from a ``.csv`` file. """
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from ta import add_all_ta_features

import matplotlib
matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt

sns.set(style="white")


def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """ Parse preprocessing arguments. """
    parser.add_argument("--drop_cols", type=str, default="")
    parser.add_argument("--file", type=str, default="file.csv")
    parser.add_argument("--output_file", type=str, default="out_file.csv")
    parser.add_argument("--graphs_path", type=str, default="graphs/")
    parser.add_argument("--sep", type=str, default="\t")
    return parser.parse_args()


def drop_cols(args: argparse.Namespace, df: pd.DataFrame, cols: str) -> pd.DataFrame:
    """ Drop the specified columns and return the resultant ``pd.DataFrame``. """
    # drop all columns specified in the comma separated list from args
    drop = [x.strip() for x in cols.split(",")]
    print("Dropping", drop)
    df = df.drop(columns=drop)

    return df


def gen_ta(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    """ Generate technical indicators and return resultant ``pd.DataFrame``. """
    print("Generating technical indicators...")
    df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)
    print("[done]")

    bad_columns = "trend_adx, trend_adx_pos, trend_adx_neg"
    df = drop_cols(args, df, bad_columns)

    return df


def cross_correlation(args: argparse.Namespace, df: pd.DataFrame) -> None:
    """ Compute cross correlation and graph as a ``.svg``. """
    GRAPHS_PATH = args.graphs_path
    assert os.path.isdir(GRAPHS_PATH)
    filename = os.path.basename(args.file)
    filename_no_ext = filename.split(".")[0]
    save_path = os.path.join(GRAPHS_PATH, filename_no_ext + ".svg")

    # Compute the correlation matrix.
    corr = df.corr()

    # Generate a mask for the upper triangle.
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure.
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap.
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    # Save to svg.
    plt.savefig(save_path)


def main() -> None:
    """ Preprocess, drop bad columns, and compute cross correlation. """
    args = parse(argparse.ArgumentParser())
    # Grab training data
    df = pd.read_csv(args.file, sep=",")
    # Generate technical indicators
    df = gen_ta(args, df)
    # Drop columns
    if args.drop_cols != "":
        df = drop_cols(args, df, args.drop_cols)

    cross_correlation(args, df)

    # Save dataframe to csv file
    df.to_csv(path_or_buf=args.output_file, sep=args.sep, index=False)


if __name__ == "__main__":
    main()
