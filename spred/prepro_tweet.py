# Copyright (c) Microsoft. All rights reserved.
import os
import json
import tqdm
import pickle
import re
import collections
import argparse
from sys import path
from data_utils.vocab import Vocabulary
from pytorch_pretrained_bert.tokenization import BertTokenizer
from data_utils.log_wrapper import create_logger
from data_utils.label_map import GLOBAL_MAP
from data_utils.glue_utils import *
DEBUG_MODE=False
MAX_SEQ_LEN = 512

logger = create_logger(__name__, to_disk=True, log_file='bert_data_proc_512.log')

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def build_data(data, dump_path, max_seq_len=MAX_SEQ_LEN, is_train=True, tokenizer=None):
    """Build data of sentence pair tasks
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = tokenizer.tokenize(sample['premise'])
            hypothesis = tokenizer.tokenize(sample['hypothesis'])
            label = sample['label']
            _truncate_seq_pair(premise, hypothesis, max_seq_len - 3)
            input_ids =tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis + ['[SEP]'] + premise + ['[SEP]'])
            type_ids = [0] * ( len(hypothesis) + 2) + [1] * (len(premise) + 1)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))

def build_qnli(data, dump_path, max_seq_len=MAX_SEQ_LEN, is_train=True, tokenizer=None):
    """Build QNLI as a pair-wise ranking task
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = tokenizer.tokenize(sample['premise'])
            hypothesis_1 = tokenizer.tokenize(sample['hypothesis'][0])
            hypothesis_2 = tokenizer.tokenize(sample['hypothesis'][1])
            label = sample['label']
            _truncate_seq_pair(premise, hypothesis_1, max_seq_len - 3)
            input_ids_1 =tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis_1 + ['[SEP]'] + premise + ['[SEP]'])
            type_ids_1 = [0] * ( len(hypothesis_1) + 2) + [1] * (len(premise) + 1)
            _truncate_seq_pair(premise, hypothesis_2, max_seq_len - 3)
            input_ids_2 = tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis_2 + ['[SEP]'] + premise + ['[SEP]'])
            type_ids_2 = [0] * ( len(hypothesis_2) + 2) + [1] * (len(premise) + 1)
            features = {'uid': ids, 'label': label, 'token_id': [input_ids_1, input_ids_2], 'type_id': [type_ids_1, type_ids_2], 'ruid':sample['ruid'], 'olabel':sample['olabel']}
            writer.write('{}\n'.format(json.dumps(features)))

def build_data_single(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
    """Build data of single sentence tasks
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = tokenizer.tokenize(sample['premise'])
            label = sample['label']
            if len(premise) >  max_seq_len - 3:
                premise = premise[:max_seq_len - 3] 
            input_ids =tokenizer.convert_tokens_to_ids(['[CLS]'] + premise + ['[SEP]'])
            type_ids = [0] * ( len(premise) + 2)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--old_glue', action='store_true', help='whether it is old GLUE, refer official GLUE webpage for details')
    args = parser.parse_args()
    return args

def main(args):
    ## hyper param
    is_old_glue = args.old_glue
    do_lower_case = args.do_lower_case
    root = args.root_dir
    assert os.path.exists(root)
    is_uncased = False
    if 'uncased' in args.bert_model:
        is_uncased = True

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=do_lower_case)

    # tweet_train_path = os.path.join(root, 'tweet/train.tsv')
    # tweet_dev_path = os.path.join(root, 'tweet/dev.tsv')
    tweet_test_path = os.path.join(root, 'tweet/test.tsv')

    # tweet_train_data = load_sst(tweet_train_path)
    # tweet_dev_data = load_sst(tweet_dev_path)
    tweet_test_data = load_sst(tweet_test_path, is_train=False)
    # logger.info('Loaded {} tweet train samples'.format(len(tweet_train_data)))
    # logger.info('Loaded {} tweet dev samples'.format(len(tweet_dev_data)))
    logger.info('Loaded {} tweet test samples'.format(len(tweet_test_data)))

    mt_dnn_suffix = 'mt_dnn'
    if is_uncased:
        mt_dnn_suffix = '{}_uncased'.format(mt_dnn_suffix)
    else:
        mt_dnn_suffix = '{}_cased'.format(mt_dnn_suffix)

    if do_lower_case:
        mt_dnn_suffix = '{}_lower'.format(mt_dnn_suffix)

    mt_dnn_root = os.path.join(root, mt_dnn_suffix)
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    # tweet_train_fout = os.path.join(mt_dnn_root, 'tweet_train.json')
    # tweet_dev_fout = os.path.join(mt_dnn_root, 'tweet_dev.json')
    tweet_test_fout = os.path.join(mt_dnn_root, 'tweet_test.json')
    # build_data_single(tweet_train_data, tweet_train_fout, tokenizer=bert_tokenizer)
    # build_data_single(tweet_dev_data, tweet_dev_fout, tokenizer=bert_tokenizer)
    build_data_single(tweet_test_data, tweet_test_fout, tokenizer=bert_tokenizer)
    logger.info('done with tweet')

if __name__ == '__main__':
    args = parse_args()
    main(args)
