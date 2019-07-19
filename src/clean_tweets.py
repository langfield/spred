import pandas as pd
import glob
import json
import os

tweetscraper_dir = '../../'
path = os.path.join(tweetscraper_dir, 'TweetScraper/Data/tweet/')
#print(path)
assert os.path.exists(path)
files = glob.glob(path + '*')
#print(files)

dictlist = []

for tweet in files:
    json_string = open(tweet, 'r').read()
    json_dict = json.loads(json_string)
    dictlist.append(json_dict)

df = pd.DataFrame(dictlist)
df = df.replace({'\n': ' '}, regex=True)
df = df.replace({'\t': ' '}, regex=True)
df = df.replace({'\r': ' '}, regex=True)
df = df.replace({'http[^\s]+': ' '}, regex=True)
df = df.replace({'bit.ly[^\s]+': ' '}, regex=True)
df = df.replace({'youtu.be[^\s]+': ' '}, regex=True)
df = df.replace({'[^\s]+.com[^\s]+': ' '}, regex=True)
df = df.replace({'\.': ' . '}, regex=True)
df = df.replace({'\(': ' ( '}, regex=True)
df = df.replace({'\)': ' ) '}, regex=True)
df = df.replace({'\)': ' ) '}, regex=True)
df = df.replace({'\?': ' ? '}, regex=True)

punctuation = ['`', '~', '!', '@', '#', '$', '%', '\^', '&', '\*', '-', '_', '\+', '=', '\[', '\]', '{', '}', '\\\\', '\|', ';', ':', '"', '\'', ',', '<', '>', '/']

for char in punctuation:
  new_string = ' ' + char + ' '
  df = df.replace({char: new_string}, regex=True)

df = df.replace({' +': ' '}, regex=True)

df['index'] = df.index
print(df.columns)
df = df.drop(labels=['ID', 'datetime', 'has_media', 'is_reply', 'is_retweet', 'medias', 'nbr_favorite', 'nbr_reply', 'nbr_retweet', 'url', 'user_id', 'usernameTweet'], axis=1)
cols = df.columns.tolist()
cols = cols[::-1]
df = df[cols]
df = df.rename(mapper={'text':'sentence'}, axis=1)
print(df.head(10))

df.to_csv("data.tsv", sep = '\t', index=False)
