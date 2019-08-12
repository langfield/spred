from eval import get_labels, model_config, data_config, train_config
from prepro_tweet import prepro_config
import argparse
import numpy as np

def score(bucket):
    '''
    Generate a score for a single bucket of data
    '''
    return np.average(bucket)

def get_scores(data):
    '''
    Return a list of score values--one per bucket of data
    '''
    # Put the data into minute-scale buckets
    prev = 0
    cur = 0
    count = 0
    bucket = []
    scores = []

    for i in range(len(data['timestamps'])):
        cur = data['timestamps'][i].minute
        while cur > prev or (cur == 0 and prev == 59):
            scores.append(score(bucket))
            bucket = []
            prev += 1
            if prev == 60:
                prev = 0
        bucket.append(data['predictions'][i])
    
    scores.append(score(np.asarray(bucket)))

    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    parser = prepro_config(parser)
    args = parser.parse_args()

    results = get_labels(args)
    print('Generating scores...')
    scores = get_scores(results)
    print(scores)