""" Preprocess a raw time-series dataset from a ``.csv`` file. """
import matplotlib

matplotlib.use("Agg")
import os
import argparse
from typing import List, Set
import numpy as np
import pandas as pd
import seaborn as sns
from ta import add_all_ta_features

# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt

sns.set(style="white")


def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """ Parse preprocessing arguments. """
    parser.add_argument("--drop_cols", type=str, default="")
    parser.add_argument("--file", type=str, default="file.csv")
    parser.add_argument("--output_file", type=str, default="out_file.csv")
    parser.add_argument("--graphs_path", type=str, default="graphs/")
    parser.add_argument("--in_sep", type=str, default=",")
    parser.add_argument("--out_sep", type=str, default="\t")
    return parser.parse_args()


def drop_cols(df: pd.DataFrame, drop: list) -> pd.DataFrame:
    """ Drop the specified columns and return the resultant ``pd.DataFrame``. """
    # drop all columns specified in the comma separated list from args
    print("Dropping", drop)
    df = df.drop(columns=drop)

    return df


def gen_ta(df: pd.DataFrame) -> pd.DataFrame:
    """ Generate technical indicators and return resultant ``pd.DataFrame``. """
    print("Generating technical indicators...")
    df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)
    print("[done]")

    bad_columns = ["trend_adx", "trend_adx_pos", "trend_adx_neg"]
    df = drop_cols(df, bad_columns)

    return df


def cross_correlation(args: argparse.Namespace, df: pd.DataFrame) -> pd.DataFrame:
    """ Compute cross correlation and graph as a ``.svg``. """
    assert os.path.isdir(args.graphs_path)
    filename = os.path.basename(args.file)
    filename_no_ext = filename.split(".")[0]
    save_path = os.path.join(args.graphs_path, filename_no_ext + ".svg")

    # correlation(df, save_path)
    corr = df.corr()

    # Generate a mask for the upper triangle.
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure.
    _f, _ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap.
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    plt.savefig(save_path)
    # Compute the correlation matrix.

    return corr


def drop_correlation(corr: pd.DataFrame) -> List[str]:
    """
    Find columns with correlation higher than ``0.5`` and return the
    column labels as a list of strings.
    """
    column_names = corr.columns[5:]
    dropped: Set[str] = set()
    n_cols = len(corr) - 5
    for i in range(n_cols):
        for j in range(n_cols):
            if i + (n_cols - j) < n_cols:
                if abs(corr.iloc[i, j]) > 0.75 and column_names[j] not in dropped:
                    dropped.add(column_names[i])
    return list(dropped)


def main() -> None:
    """ Preprocess, drop bad columns, and compute cross correlation. """
    args = parse(argparse.ArgumentParser())
    # Grab training data
    df = pd.read_csv(args.file, sep=args.in_sep)
    # Generate technical indicators
    df = gen_ta(df)
    # Drop columns
    if args.drop_cols != "":
        drop = [x.strip() for x in args.drop_cols.split(args.in_sep)]
        df = drop_cols(df, drop)

    corr = cross_correlation(args, df)
    corr_cols = drop_correlation(corr)
    df = drop_cols(df, corr_cols)

    # Save dataframe to csv file
    df.to_csv(path_or_buf=args.output_file, sep=args.out_sep, index=False)


if __name__ == "__main__":
    main()
