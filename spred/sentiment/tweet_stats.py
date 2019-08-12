from clean_tweets import get_df
import pandas
import matplotlib.pyplot as plt
from datetimerange import DateTimeRange
from dateutil.relativedelta import relativedelta


def get_minute_counts():
    print('Loading data...')
    df = get_df('../../../TweetScraper/Data/tweet/')
    print('Number of data points:', df.shape[0])

    # Put the data into minute-scale buckets
    minute_counts = []
    prev = 0
    cur = 0
    count = 0
    for i, v in df.iterrows():
        cur = v['datetime'].minute
        while cur > prev or (cur == 0 and prev == 59):
            minute_counts.append(count)
            count = 0
            prev += 1
            if prev == 60:
                prev = 0
        count += 1

    return minute_counts, df

def plot_tweet_volume(minute_counts, df):
    time_range = DateTimeRange(df['datetime'][df.index[0]], df['datetime'][df.index[-1]])
    times = []
    delta = 2
    for value in time_range.range(relativedelta(hours=+delta)):
        times.append(str(value))
    fig, ax = plt.subplots()
    ax.plot(minute_counts)
    ax.set_title('Ethereum Tweet Volume')
    ax.set_ylabel('Number of Tweets')
    ax.set_xlabel('Time')
    ax.set_xticks(range(0, len(minute_counts), 60 * delta))
    ax.set_xticklabels(times, rotation=45, ha='right')
    plt.show()

if __name__ == "__main__":
    # plot tweets per minute
    minute_counts, df = get_minute_counts()
    plot_tweet_volume(minute_counts, df)