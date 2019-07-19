import pandas as pd
import glob
import json

files = glob.glob('TweetScraper/Data/tweet/*')

dictlist = []

for tweet in files:
    json_string = open(tweet, 'r').read()
    json_dict = json.loads(json_string)
    dictlist.append(json_dict)

df = pd.DataFrame(dictlist)
df = df.replace({'\n': ' '}, regex=True)
df = df.replace({'\t': ' '}, regex=True)
df = df.replace({'\r': ' '}, regex=True)

df.to_csv("data.tsv", sep = '\t')
