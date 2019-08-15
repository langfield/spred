import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--drop_cols", type=str, default="")
parser.add_argument("--file", type=str, default="file")
parser.add_argument("--output_file", type=str, default="file")
args = parser.parse_args()

# Grab training data.
raw_data = pd.read_csv(args.file, sep=',')
# drop all columns specified in the comma separated list from args
drop = [x.strip() for x in args.drop_cols.split(',')]
print("Dropping", drop)
if len(drop) > 0:
    raw_data = raw_data.drop(columns=drop)

print()
print(raw_data.head())
print("\nSaving to", args.output_file)
raw_data.to_csv(path_or_buf=args.output_file, sep="\t", index=False)