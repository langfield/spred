import argparse
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="file")
parser.add_argument("--output_file", type=str, default="file")
parser.add_argument("--sep", type=str, default="\t")
args = parser.parse_args()

# Grab training data.
raw_data = pd.read_csv(args.file, sep=args.sep)

print("Generating technical indicators...")
raw_data = add_all_ta_features(raw_data, 
                               "Open", 
                               "High", 
                               "Low", 
                               "Close", 
                               "Volume", 
                               fillna=True)
#raw_data = dropna(raw_data)
print("[done]")

print(raw_data.head())
print("\nSaving to", args.output_file)
raw_data.to_csv(path_or_buf=args.output_file, sep="\t", index=False)