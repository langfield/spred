# Copyright (c) Microsoft. All rights reserved.
import os
import json
import tqdm
import pickle
import re
import collections
import argparse
import pandas
from sys import path

path.insert(1, "../../../mt-dnn/")
from data_utils.vocab import Vocabulary
from pytorch_pretrained_bert.tokenization import BertTokenizer
from data_utils.log_wrapper import create_logger
from data_utils.label_map import GLOBAL_MAP
from clean_tweets import get_df

DEBUG_MODE = False
MAX_SEQ_LEN = 512

logger = create_logger(__name__, to_disk=True, log_file="bert_data_proc_512.log")


def build_data_single(data, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
    """Build data of single sentence tasks
    """
    result = []
    for idx, sample in enumerate(data):
        ids = sample["uid"]
        premise = tokenizer.tokenize(sample["premise"])
        label = sample["label"]
        if len(premise) > max_seq_len - 3:
            premise = premise[: max_seq_len - 3]
        input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + premise + ["[SEP]"])
        type_ids = [0] * (len(premise) + 2)
        features = {
            "uid": ids,
            "label": label,
            "token_id": input_ids,
            "type_id": type_ids,
            "factor": 1.0,
            "timestamp": sample["timestamp"],
        }
        result.append(features)
    return result


def prepro_config(parser):
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument(
        "--tweet_data_dir", type=str, default="../../../TweetScraper/Data/tweet/"
    )
    return parser


def load_sst(df):
    rows = []
    for index, row in df.iterrows():
        sample = {
            "uid": int(row["index"]),
            "premise": row["sentence"],
            "timestamp": row["datetime"],
            "label": 0,
        }

        rows.append(sample)
    return rows


def get_prepro_data(args):
    ## hyper param
    do_lower_case = args.do_lower_case
    is_uncased = False
    if "uncased" in args.model:
        is_uncased = True

    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=do_lower_case)

    df = get_df(args.tweet_data_dir)

    tweet_test_data = load_sst(df)
    logger.info("Loaded {} tweet test samples".format(len(tweet_test_data)))

    mt_dnn_suffix = "mt_dnn"
    if is_uncased:
        mt_dnn_suffix = "{}_uncased".format(mt_dnn_suffix)
    else:
        mt_dnn_suffix = "{}_cased".format(mt_dnn_suffix)

    if do_lower_case:
        mt_dnn_suffix = "{}_lower".format(mt_dnn_suffix)

    data = build_data_single(tweet_test_data, tokenizer=tokenizer)
    logger.info("done with tweet")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = prepro_config(parser)
    args = parser.parse_args()
    get_prepro_data(args)
