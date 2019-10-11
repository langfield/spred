""" Preprocesses a raw time-series dataset from a ``.csv`` file. """
import os
import argparse
from typing import List, Set

# pylint: disable=wrong-import-position, ungrouped-imports, bad-continuation
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ta import add_all_ta_features

sns.set(style="white")

CORR_THRESHOLD = 0.75
KEPT_IND = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "volume_em",
    "volume_vpt",
    "volume_nvi",
    "volatility_atr",
    "volatility_bbh",
    "volatility_bbl",
    "volatility_bbm",
    "volatility_bbhi",
    "volatility_bbli",
    "volatility_kchi",
    "volatility_kcli",
    "volatility_dcli",
    "trend_macd",
    "trend_ema_fast",
    "trend_ema_slow",
    "trend_dpo",
    "trend_kst",
    "trend_kst_diff",
    "trend_ichimoku_a",
    "trend_ichimoku_b",
    "trend_visual_ichimoku_b",
    "trend_aroon_up",
    "momentum_mfi",
    "momentum_stoch",
    "momentum_wr",
    "momentum_ao",
    "others_dr",
    "others_cr",
]


def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse preprocessing arguments.

    Parameters
    ----------
    parser : ``argparse.ArgumentParser``, required.
        The parser to add arguments.

    Returns
    -------
    args : ``argparse.Namespace``.
        The resultant arguments object.
    """
    parser.add_argument("--drop_cols", type=str, default="")
    parser.add_argument("--file", type=str, default="file.csv")
    parser.add_argument("--output_file", type=str, default="out_file.csv")
    parser.add_argument("--graphs_path", type=str, default="graphs/")
    parser.add_argument("--in_sep", type=str, default=",")
    parser.add_argument("--out_sep", type=str, default="\t")
    parser.add_argument("--correlation", action="store_true")
    args = parser.parse_args()
    return args


def gen_ta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate technical indicators and return resultant ``pd.DataFrame``.

    Parameters
    ----------
    df : ``pd.DataFrame``, required.
        Time series data for which we wish to generate technical indicators.

    Returns
    -------
    df : ``pd.DataFrame``.
        The same as the input, but with TI columns added.
    """
    print("Generating technical indicators...")
    df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)
    print("[done]")

    bad_columns = ["trend_adx", "trend_adx_pos", "trend_adx_neg"]
    df = df.drop(columns=bad_columns)

    return df


def cross_correlation(
    df: pd.DataFrame, graphs_path: str, data_path: str
) -> pd.DataFrame:
    """
    Compute cross correlation and graph as a ``.svg``.

    Parameters
    ----------
    df : ``pd.DataFrame``, required.
        The data for which we wish to compute cross correlation.
    graphs_path : ``str``, required.
        The path to the directory to save graphs.
    data_path : ``str``, required.
        The path to the source for ``df``.

    Returns
    -------
    corr : ``pd.DataFrame``.
        The correlation matrix for all columns in ``df``.
    """
    assert os.path.isdir(graphs_path)
    filename = os.path.basename(data_path)
    filename_no_ext = filename.split(".")[0]
    save_path = os.path.join(graphs_path, filename_no_ext + ".svg")

    corr = df.corr()

    # Generate a mask for the upper triangle.
    # pylint: disable=unsupported-assignment-operation
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure.
    _f, _ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap.
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio.
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
    return corr


def drop_correlation(corr: pd.DataFrame) -> List[str]:
    """
    Find columns with correlation higher than ``CORR_THRESHOLD``
    and return the column labels as a list of strings.

    Parameters
    ----------
    corr : ``pd.DataFrame``, required.
        Correlation matrix.

    Returns
    -------
    cols_to_drop : ``List[str]``.
        Columns to drop, i.e. those with corr. greater than ``CORR_THRESHOLD``.
    """
    column_names = corr.columns[5:]
    drop: Set[str] = set()
    n_cols = len(corr) - 5
    for i in range(n_cols):
        for j in range(n_cols):
            if i + (n_cols - j) < n_cols:
                if (
                    abs(corr.iloc[i, j]) > CORR_THRESHOLD
                    and column_names[j] not in drop
                ):
                    drop.add(column_names[i])
    cols_to_drop = list(drop)
    return cols_to_drop


def main() -> None:
    """ Preprocess, drop bad columns, and compute cross correlation. """
    args = parse(argparse.ArgumentParser())
    # Grab training data.
    df = pd.read_csv(args.file, sep=args.in_sep)
    # Generate technical indicators.
    df = gen_ta(df)
    # Drop columns.
    if args.drop_cols != "":
        drop = [x.strip() for x in args.drop_cols.split(args.in_sep)]
        df = df.drop(columns=drop)

    corr = cross_correlation(df, args.graphs_path, args.file)
    if args.correlation:
        corr_cols = drop_correlation(corr)
    else:
        corr_cols = [s for s in df.columns if s not in KEPT_IND]
    df = df.drop(columns=corr_cols)

    # Save dataframe to ``.csv`` file.
    df.to_csv(path_or_buf=args.output_file, sep=args.out_sep, index=False)


if __name__ == "__main__":
    main()
