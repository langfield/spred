import pandas as pd

df = pd.read_csv("gemini_ETHUSD_2019_1min.csv")
print(df.head(10))
df = df.drop(['Unix Timestamp', 'Date', 'Symbol'], axis=1)
print(df.head(10))

df.to_csv(path_or_buf="gemini.csv", sep="\t", index=False)
