# Copyright (c) Microsoft. All rights reserved.
import argparse
import json
import os
import random
import numpy as np
import torch

import sys

sys.path.insert(1, "../../../mt-dnn/")
from pytorch_pretrained_bert.modeling import BertConfig
from data_utils.glue_utils import eval_model
from data_utils.label_map import (
    DATA_META,
    GLOBAL_MAP,
    DATA_TYPE,
    DATA_SWAP,
    TASK_TYPE,
    generate_decoder_opt,
)
from data_utils.log_wrapper import create_logger
from data_utils.utils import set_environment
from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDNNModel

from prepro_tweet import get_prepro_data
from prepro_tweet import prepro_config

from datetime import datetime


def model_config(parser):
    parser.add_argument("--mtl_opt", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0)
    parser.add_argument("--max_seq_len", type=int, default=512)
    return parser


def data_config(parser):
    parser.add_argument(
        "--log_file", default="mt-dnn-train.log", help="path for log file."
    )
    parser.add_argument(
        "--init_checkpoint", default="mt_dnn_models/mt_dnn_base_uncased.pt", type=str
    )
    parser.add_argument("--data_sort_on", action="store_true")
    parser.add_argument("--name", default="farmer")
    return parser


def train_config(parser):
    parser.add_argument(
        "--cuda",
        type=bool,
        default=torch.cuda.is_available(),
        help="whether to use GPU acceleration.",
    )
    parser.add_argument("--log_per_updates", type=int, default=500)
    parser.add_argument("--batch_size_eval", type=int, default=8)
    parser.add_argument("--output_dir", default="checkpoint")
    parser.add_argument(
        "--seed",
        type=int,
        default=2018,
        help="random seed for data shuffling, embedding init, etc.",
    )
    parser.add_argument(
        "--task_config_path", type=str, default="configs/tasks_config.json"
    )

    return parser


def submit(path, data):
    header = "index\ttime\tprediction"
    with open(path, "w") as writer:
        predictions = data["predictions"]
        uids = data["uids"]
        timestamps = data["timestamps"]
        writer.write("{}\n".format(header))
        assert len(predictions) == len(uids)
        assert len(timestamps) == len(timestamps)
        # sort label
        output_tuples = [
            (int(uids[i]), predictions[i], timestamps[i]) for i in range(len(uids))
        ]
        for uid, pred, time in output_tuples:
            writer.write("{}\t{}\t{}\n".format(uid, time, pred))


def run_eval(tweet_data, logger, args):
    logger.info("Launching the MT-DNN evaluation")
    opt = vars(args)
    # update data dir
    tasks = {}
    tasks_class = {}
    nclass_list = []

    prefix = "sst"
    assert prefix in DATA_META
    assert prefix in DATA_TYPE
    data_type = DATA_TYPE[prefix]
    nclass = DATA_META[prefix]
    task_id = len(tasks)

    if prefix not in tasks:
        tasks[prefix] = len(tasks)
        if args.mtl_opt < 1:
            nclass_list.append(nclass)

    task_id = tasks_class[DATA_META[prefix]] if args.mtl_opt > 0 else tasks[prefix]
    task_type = TASK_TYPE[prefix]

    assert prefix in DATA_TYPE
    data_type = DATA_TYPE[prefix]

    timestamp_list = []
    for tweet in tweet_data:
        timestamp = tweet["timestamp"]
        timestamp_list.append(timestamp)

    test_data = None
    test_data = BatchGen(
        tweet_data,
        batch_size=args.batch_size_eval,
        gpu=args.cuda,
        is_train=False,
        task_id=task_id,
        maxlen=args.max_seq_len,
        pairwise=None,
        data_type=data_type,
        task_type=task_type,
    )

    model_path = os.path.abspath(args.init_checkpoint)
    state_dict = None

    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        config = state_dict["config"]
        opt.update(config)
    else:
        logger.error("#" * 20)
        logger.error(
            "Could not find the init model!\n The parameters will be initialized randomly!"
        )
        logger.error("#" * 20)
        config = BertConfig(vocab_size_or_config_json_file=30522).to_dict()
        opt.update(config)

    model = MTDNNModel(opt, state_dict=state_dict)

    logger.info("Total number of params: {}".format(model.total_param))

    if args.cuda:
        model.cuda()

    # test eval
    if test_data is not None:
        test_metrics, test_predictions, scores, golds, test_ids = eval_model(
            model, test_data, dataset=prefix, use_cuda=args.cuda, with_label=False
        )
        results = {
            "metrics": test_metrics,
            "predictions": test_predictions,
            "uids": test_ids,
            "scores": scores,
            "timestamps": timestamp_list,
        }
        return results

    return None


def get_labels(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.abspath(output_dir)

    set_environment(args.seed, args.cuda)
    log_path = args.log_file
    logger = create_logger(__name__, to_disk=True, log_file=log_path)

    tasks_config = {}
    if os.path.exists(args.task_config_path):
        with open(args.task_config_path, "r") as reader:
            tasks_config = json.loads(reader.read())

    # Run preprocessing script.
    data = get_prepro_data(args)
    # data = [{'uid': 0, 'label': 0, 'token_id': [101, 100, 3114, 2000, 2224, 100, 100, 100, 1585, 100, 2012, 100, 100, 2031, 2764, 1037, 6028, 2005, 7861, 8270, 4667, 2951, 1999, 2189, 1998, 23820, 2009, 2000, 1037, 26381, 1011, 17727, 2121, 3401, 13876, 7028, 2000, 1996, 2529, 4540, 1012, 27593, 1012, 1048, 2100, 1013, 100, 1002, 102], 'type_id': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'factor': 1.0, 'timestamp': datetime(2019,8,9,20,0,0)}]
    return run_eval(data, logger, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    parser = prepro_config(parser)
    args = parser.parse_args()

    results = get_labels(args)
    official_score_file = os.path.join(output_dir, "tweet_test_scores.tsv")
    submit(official_score_file, results)
    logger.info("[new test scores saved.]")
