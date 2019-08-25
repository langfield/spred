import argparse
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna

def parse(parser):
    parser.add_argument("--drop_cols", type=str, default="")
    parser.add_argument("--file", type=str, default="file.csv")
    parser.add_argument("--output_file", type=str, default="out_file.csv")
    parser.add_argument("--sep", type=str, default="\t")
    return parser.parse_args()

def drop_cols(df, cols):
    # drop all columns specified in the comma separated list from args
    drop = [x.strip() for x in args.drop_cols.split(",")]
    print("Dropping", drop)
    df = raw_data.drop(columns=drop)

    return df

def gen_ta(df):
    print("Generating technical indicators...")
    df = add_all_ta_features(
        df, "Open", "High", "Low", "Close", "Volume", fillna=True
    )
    print("[done]")

    return df

def main():
    args = parse(argparse.ArgumentParser())
    # Grab training data
    df = pd.read_csv(args.file, sep=",")
    # Generate technical indicators
    df = gen_ta(df)
    # Drop columns
    if args.drop_cols == "":
        args.drop_cols = "trend_adx, trend_adx_pos, trend_adx_neg"
        df = drop_cols(df, args.drop_cols)

    # Save dataframe to csv file
    df.to_csv(path_or_buf=args.output_file, sep=args.sep, index=False)

if __name__ = "__main__":
    main()