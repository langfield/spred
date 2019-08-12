# Copyright (c) Microsoft. All rights reserved.
import argparse
import json
import os
import random
import numpy as np
import torch

import sys
sys.path.insert(1, '../../../mt-dnn/')
from pytorch_pretrained_bert.modeling import BertConfig
from data_utils.glue_utils import submit, eval_model
from data_utils.label_map import DATA_META, GLOBAL_MAP, DATA_TYPE, DATA_SWAP, TASK_TYPE, generate_decoder_opt
from data_utils.log_wrapper import create_logger
from data_utils.utils import set_environment
from mt_dnn.batcher import BatchGen
from mt_dnn.model import MTDNNModel

def model_config(parser):
    parser.add_argument('--answer_opt', type=int, default=0, help='0,1')
    parser.add_argument('--mtl_opt', type=int, default=0)
    parser.add_argument('--ratio', type=float, default=0)
    parser.add_argument('--max_seq_len', type=int, default=512)
    return parser

def data_config(parser):
    parser.add_argument('--log_file', default='mt-dnn-train.log', help='path for log file.')
    parser.add_argument("--init_checkpoint", default='mt_dnn_models/mt_dnn_base_uncased.pt', type=str)
    parser.add_argument('--data_dir', default='data/mt_dnn_uncased_lower')
    parser.add_argument('--data_sort_on', action='store_true')
    parser.add_argument('--name', default='farmer')
    parser.add_argument('--test_datasets', default='sst')
    return parser

def train_config(parser):
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    parser.add_argument('--log_per_updates', type=int, default=500)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--output_dir', default='checkpoint')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--task_config_path', type=str, default='configs/tasks_config.json')

    return parser

parser = argparse.ArgumentParser()
parser = data_config(parser)
parser = model_config(parser)
parser = train_config(parser)
args = parser.parse_args()

output_dir = args.output_dir
data_dir = args.data_dir
args.test_datasets = args.test_datasets.split(',')
os.makedirs(output_dir, exist_ok=True)
output_dir = os.path.abspath(output_dir)

set_environment(args.seed, args.cuda)
log_path = args.log_file
logger =  create_logger(__name__, to_disk=True, log_file=log_path)
logger.info(args.answer_opt)

tasks_config = {}
if os.path.exists(args.task_config_path):
    with open(args.task_config_path, 'r') as reader:
        tasks_config = json.loads(reader.read())

def dump(path, data):
    with open(path ,'w') as f:
        json.dump(data, f)

def main():
    logger.info('Launching the MT-DNN evaluation')
    opt = vars(args)
    # update data dir
    opt['data_dir'] = data_dir
    tasks = {}
    tasks_class = {}
    nclass_list = []
    decoder_opts = []
    dropout_list = []

    opt['answer_opt'] = decoder_opts
    opt['tasks_dropout_p'] = dropout_list

    test_data_list = []
    for dataset in args.test_datasets:
        prefix = dataset.split('_')[0]
        if prefix in tasks: continue
        assert prefix in DATA_META
        assert prefix in DATA_TYPE
        data_type = DATA_TYPE[prefix]
        nclass = DATA_META[prefix]
        task_id = len(tasks)

        if prefix not in tasks:
            tasks[prefix] = len(tasks)
            if args.mtl_opt < 1: nclass_list.append(nclass)

        task_id = tasks_class[DATA_META[prefix]] if args.mtl_opt > 0 else tasks[prefix]
        task_type = TASK_TYPE[prefix]

        assert prefix in DATA_TYPE
        data_type = DATA_TYPE[prefix]

        test_path = os.path.join(data_dir, '{}_test.json'.format(dataset))
        test_data = None
        if os.path.exists(test_path):
            test_data = BatchGen(BatchGen.load(test_path, False, pairwise=None, maxlen=args.max_seq_len),
                                  batch_size=args.batch_size_eval,
                                  gpu=args.cuda, is_train=False,
                                  task_id=task_id,
                                  maxlen=args.max_seq_len,
                                  pairwise=None,
                                  data_type=data_type,
                                  task_type=task_type)
        test_data_list.append(test_data)

    model_path = os.path.abspath(args.init_checkpoint)
    state_dict = None

    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        config = state_dict['config']
        print(opt)
        opt.update(config)
        print(opt)
    else:
        logger.error('#' * 20)
        logger.error('Could not find the init model!\n The parameters will be initialized randomly!')
        logger.error('#' * 20)
        config = BertConfig(vocab_size_or_config_json_file=30522).to_dict()
        opt.update(config)

    print(opt)
    model = MTDNNModel(opt, state_dict=state_dict)

    logger.info("Total number of params: {}".format(model.total_param))

    if args.cuda:
        model.cuda()

    for idx, dataset in enumerate(args.test_datasets):
        prefix = dataset.split('_')[0]
        label_dict = GLOBAL_MAP.get(prefix, None)
        
        # test eval
        test_data = test_data_list[idx]
        if test_data is not None:
            test_metrics, test_predictions, scores, golds, test_ids= eval_model(model, 
                                                                                test_data,
                                                                                dataset=prefix,
                                                                                use_cuda=args.cuda, 
                                                                                with_label=False)
            score_file = os.path.join(output_dir, '{}_test_scores.json'.format(dataset))
            results = {'metrics': test_metrics, 
                       'predictions': test_predictions, 
                       'uids': test_ids, 
                       'scores': scores}
            dump(score_file, results)
            official_score_file = os.path.join(output_dir, 
                                               '{}_test_scores.tsv'.format(dataset))
            submit(official_score_file, results, label_dict)
            logger.info('[new test scores saved.]')


if __name__ == '__main__':
    main()
