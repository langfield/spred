# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XLSpred finetuning runner."""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os
import sys
import random
import pandas as pd
from io import open

import numpy as np
import torch
from perm_generator import _local_perm
from split_a_and_b_torch import _split_a_and_b
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.modeling_xlnet import XLNetModel, XLNetConfig
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from sample_mask_spred import _sample_mask

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


UNK_ID = -9996
SEP_ID = -9997
CLS_ID = -9998
MASK_ID = -9999
NUM_PREDICT = 12345

class XLSpredDataset(Dataset):
    def __init__(self, 
                 corpus_path, 
                 seq_len, 
                 num_predict,
                 data_batch_size,
                 reuse_len,
                 encoding="utf-8", 
                 corpus_lines=None, 
                 on_memory=True):

        self.seq_len = seq_len
        self.num_predict = num_predict
        self.data_batch_size = data_batch_size
        self.reuse_len = reuse_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.sample_counter = 0

        # Load samples into memory from file.
        self.raw_data = pd.read_csv(corpus_path)

        # Add and adjust columns.
        self.raw_data["Average"] = (self.raw_data["High"] + self.raw_data["Low"])/2
        self.raw_data['Volume'] = self.raw_data['Volume'] + 0.000001 # Avoid NaNs
        self.raw_data["Average_ld"] = (np.log(self.raw_data['Average']) - 
                                    np.log(self.raw_data['Average']).shift(1))
        self.raw_data["Volume_ld"] = (np.log(self.raw_data['Volume']) - 
                                   np.log(self.raw_data['Volume']).shift(1))
        self.raw_data = self.raw_data[1:]

        # convert data to tensor of shape(rows, features)
        self.tensor_data = torch.tensor(self.raw_data.iloc[:,[7,8]].values)
        self.features = self.create_features(self.tensor_data)
        print('len of features', len(self.features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    def create_features(self, tensor_data):
        """
        Returns a list of features of the form 
        (input, input_raw, is_masked, target, seg_id, label).
        """
        original_data_len = self.tensor_data.shape[0]
        seq_len = self.seq_len
        num_predict = self.num_predict
        batch_size = self.data_batch_size
        reuse_len = self.reuse_len

        print('original_data_len', original_data_len)
        print('seq_len', seq_len)
        print('reuse_len', reuse_len)

        # batchify the tensor as done in original xlnet implementation
        # This splits our data into shape(batch_size, data_len)
        # NOTE: data holds indices--not raw data
        # TODO: Add ``bi_data`` block from ``data_utils.py``. 
        data = torch.tensor(batchify(np.arange(0, original_data_len), batch_size))
        data_len = data.shape[1]
        sep_array = torch.tensor(np.array([SEP_ID], dtype=np.int64))
        cls_array = torch.tensor(np.array([CLS_ID], dtype=np.int64))

        i = 0
        features = []
        while i + seq_len <= data_len:
            # TODO: Is ``all_ok`` supposed to be inside or outside outer loop?
            all_ok = True
            for idx in range(batch_size):
                inp = data[idx, i: i + reuse_len]
                tgt = data[idx, i + 1: i + reuse_len + 1]
                results = _split_a_and_b(
                    data[idx],
                    begin_idx=i + reuse_len,
                    tot_len=seq_len - reuse_len - 3,
                    extend_target=True)
                if results is None:
                    all_ok = False
                    break

                # unpack the results
                (a_data, b_data, label, _, a_target, b_target) = tuple(results)
                
                # sample ngram spans to predict
                # TODO: Add ``bi_data`` stuff above. 
                bi_data = False
                reverse = bi_data and (idx // (bsz_per_core // 2)) % 2 == 1

                # TODO: Pass in ``num_predict`` as an argument or class var?
                num_predict_1 = num_predict // 2
                num_predict_0 = num_predict - num_predict_1
               
                print("inp shape:", inp.shape)
                print("num_predict_0:", num_predict_0) 
                mask_0 = _sample_mask(inp,
                                      reverse=reverse,
                                      goal_num_predict=num_predict_0)
                mask_1 = _sample_mask(torch.cat([a_data,
                                                 sep_array,
                                                 b_data,
                                                 sep_array,
                                                 cls_array]),
                                      reverse=reverse, 
                                      goal_num_predict=num_predict_1)

                # concatenate data
                cat_data = torch.cat([inp, a_data, sep_array, b_data,
                                            sep_array, cls_array])
                seg_id = torch.tensor([0] * (reuse_len + a_data.shape[0]) + [0] +
                                      [1] * b_data.shape[0] + [1] + [2])
                print("mask_0 shape:", mask_0.shape)
                # TODO: Should these even be here?
                assert cat_data.shape[0] == seq_len
                assert mask_0.shape[0] == seq_len // 2
                assert mask_1.shape[0] == seq_len // 2

                # the last two CLS's are not used, just for padding purposes
                tgt = torch.cat([tgt, a_target, b_target, cls_array, cls_array])
                assert tgt.shape[0] == seq_len

                mask_0 = torch.Tensor(mask_0)
                mask_1 = torch.Tensor(mask_1)
                is_masked = torch.cat([mask_0, mask_1], 0)
                print('cat_data', cat_data)
                print('is_masked', is_masked)
                """
                We append a vector of NaNs to tensor_data to serve as our ``[SEP]``, ``[CLS]`` vector.
                So ``mod_tensor_data`` is just ``tensor_data`` with this NaN vector added to the end. 
                ``zeroed_cat_data`` only modifies indices less than zero, and changes them to point to
                the NaN vector in ``mod_tensor_data``. Thus ``input_raw`` is the raw data with functional
                token indices yielding the NaN vector. 
                """
                dim = tensor_data.shape[-1]
                nan_tensor = torch.Tensor([[0] * dim]).double()
                mod_tensor_data = torch.cat([tensor_data, nan_tensor])
                nan_index = len(mod_tensor_data) - 1 
                zeroed_cat_data = torch.Tensor([nan_index if index < 0 else index for index in cat_data]).long() 
                print("type of ``zeroed_cat_data``:", type(zeroed_cat_data))
                print("type of ``zeroed_cat_data[0]``:", type(zeroed_cat_data[0]))
                print("``zeroed_cat_data[0]``:", zeroed_cat_data[0])
                input_raw = mod_tensor_data[zeroed_cat_data]
                
                features.append((cat_data, input_raw, is_masked, tgt, seg_id, label))
                
            if not all_ok:
                break

            i += reuse_len
        
        return features


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, sample, lm_labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            sample: tensor. The sample sequence.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.sample = sample
        self.lm_labels = lm_labels  # masked words for language model

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, lm_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.lm_label_ids = lm_label_ids

def batchify(data, batch_size):
    num_step = len(data) // batch_size
    data = data[:batch_size * num_step]
    data = data.reshape(batch_size, num_step)

    return data

def random_word(tokens, num_rows):
    """
    Masking some random tokens for Language Model task with probabilities 
    as in the original BERT paper.
    :param tokens: list of row indices for sequence data.
    :param num_rows: number of rows in the entire dataset. 
    :return: list of masked row indices, list of unmasked row indices
    """

    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = MASK_ID

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.randrange(num_rows)

            # -> rest 10% randomly keep current token

            # append the unmasked token to output_labels
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

def convert_example_to_features(example, max_seq_length, num_rows):
    """
    Convert a raw sample into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing row indices of sequence input
    :param max_seq_length: int, maximum length of sequence.
    :param num_rows: int, number of rows in the dataset.
    :return: InputFeatures, containing all inputs and labels of one sample as 
    IDs (as used for model training)
        (input_ids, input_mask, lm_label_ids)
        input_ids: list of masked row indices
        input_mask: masks out padding
        lm_label_ids: list of unmasked row indices
    """

    # Mask random tokens (words) in the sequences and return the corresponding
    # label (indices) list 
    input_ids, lm_label_ids = random_word(example.sample, num_rows)
    # account for CLS, SEP 
    # a label is an index for a vocab word
    input_ids = ([CLS_ID] + input_ids + [SEP_ID])
    lm_label_ids = ([-1] + lm_label_ids + [-1])

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    # if example.guid < 5:
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % (example.guid))
    #     logger.info("tokens: %s" % " ".join(
    #             [str(x) for x in tokens]))
    #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #     logger.info(
    #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #     logger.info("LM label: %s " % (lm_label_ids))
    #     logger.info("Is next sentence label: %s " % (example.is_next))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             lm_label_ids=lm_label_ids)
    return features


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_corpus",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.")
    parser.add_argument("--xlspred_model", default=None, type=str, required=True,
                        help="XLSpred pre-trained model path")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--data_batch_size",
                        default=1,
                        type=int,
                        help="Batch size for data loading.")
    parser.add_argument("--adam_epsilon", 
                        default=1e-8, 
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", 
                        default=0, 
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--on_memory",
                        action='store_true',
                        help="Whether to load train samples into memory or use disk")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_predict',
                        type=int,
                        default=32,
                        help="number of tokens to predict in a sequence")
    parser.add_argument('--reuse_len',
                        type=int,
                        default=64,
                        help="amount of context to reuse for recurrent memory")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type = float, default = 0,
                        help = "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train:
        raise ValueError("Training is currently the only implemented execution option. Please set `do_train`.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and ( n_gpu > 1 and torch.distributed.get_rank() == 0  or n_gpu <=1 ):
        os.makedirs(args.output_dir)

    #train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        print("Loading Train Dataset", args.train_corpus)
        train_dataset = XLSpredDataset(args.train_corpus, 
                                       seq_len=args.max_seq_length,
                                       num_predict=args.num_predict,
                                       data_batch_size=args.data_batch_size,
                                       reuse_len=args.reuse_len,
                                       corpus_lines=None, 
                                       on_memory=args.on_memory) 
        num_train_optimization_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    config = XLNetConfig.from_pretrained(args.xlspred_model)
    model = XLNetModel(config)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=args.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            #TODO: check if this works with current data generator from disk that relies on next(file)
            # (it doesn't return item back by index)
            train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        print('batch', args.train_batch_size)
        model.train()        
        seq_len = args.max_seq_length
        reuse_len = args.reuse_len
        non_reuse_len = seq_len - reuse_len
        
        # TODO: how should this be set?
        perm_size = seq_len // 2
        assert perm_size <= reuse_len and perm_size <= non_reuse_len

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)

                print("\n\n =========================")
                print("Batch length:", len(batch))
                # print("Batch contents:", batch)
                print("batch[0]:", batch[0])
                print("len(batch[0]):", len(batch[0]))
                print("input.shape:", batch[0].shape)
                print("is_masked.shape:", batch[1].shape)
                print("target.shape:", batch[2].shape)
                print("seg_id.shape:", batch[3].shape)
                print("label.shape:", batch[4].shape)
                inputs, inputs_raw, is_maskeds, targets, seg_ids, labels = batch
                # We use `input_ids`, `input_mask`, and `lm_label_ids` as arguments for
                # `perm_generator_torch` function, which yields `perm_mask` and `target_mapping`.  
                # 
                # ARGUMENT MAPPING FOR `perm_generator_torch`. 
                #   `input_ids` --> `inputs`
                #   `lm_label_ids` --> `targets`
                #   `input_mask` --> `is_masked`
                #=======PERM GENERATOR========
                
                perm_mask = []
                new_targets = []
                target_mask = []
                target_mappings = []
                # loop over batches
                for idx in range(len(batch[0])):
                    input_row = inputs[idx]
                    
                    perm_0 = _local_perm(input_row[:reuse_len], 
                                           targets[idx][:reuse_len], 
                                           is_maskeds[idx][:reuse_len].byte(),
                                           perm_size,
                                           reuse_len,
                                           device)
                    perm_mask_0, target_0, target_mask_0, _, _ = perm_0
                    
                    perm_1 = _local_perm(input_row[reuse_len:], 
                                           targets[idx][reuse_len:], 
                                           is_maskeds[idx][reuse_len:].byte(),
                                           perm_size,
                                           non_reuse_len,
                                           device)
                    perm_mask_1, target_1, target_mask_1, _, _ = perm_1

                    perm_mask_0 = torch.cat([perm_mask_0, torch.ones([reuse_len, non_reuse_len])], 
                                            dim=1)
                    
                    perm_mask_1 = torch.cat([torch.zeros([non_reuse_len, reuse_len]), perm_mask_1],
                                            dim=1)
                    
                    perm_mask.append(torch.cat([perm_mask_0, perm_mask_1], dim=0))
                    new_targets.append(torch.cat([target_0, target_1], dim=0))
                    target_mask_row = torch.cat([target_mask_0, target_mask_1], dim=0)
                    target_mask.append(target_mask_row)
                    # TODO: we are currently excluding input_k and input_q

                    indices = torch.arange(0, seq_len)
                    bool_target_mask = target_mask_row.byte()
                    # Has length equal to num `True` vals in `bool_target_mask` : <= seq_len
                    indices = indices[bool_target_mask] 
                    # length of indices after removing the masked out tokens
                    index_len = indices.shape[0]

                    # extra padding due to CLS/SEP introduced after prepro
                    actual_num_predict = indices.shape[0]
                    pad_len = seq_len - actual_num_predict

                    # target mapping
                    inp = indices % seq_len
                    inp_ = torch.unsqueeze(inp, 1)
                    target_mapping = torch.FloatTensor(index_len, seq_len).zero_()
                    target_mapping.scatter_(1, inp_, 1) # Shape: (actual_num_predict, seq_len)
                    paddings = torch.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
                    target_mapping = torch.cat([target_mapping, paddings])
                    target_mappings.append(target_mapping)

                perm_mask = torch.stack(perm_mask)
                new_targets = torch.stack(new_targets)
                target_mask = torch.stack(target_mask)
                # Shape: (bsz, actual_num_predict, seq_len)
                target_mappings = torch.stack(target_mappings) 

                #=======PERM GENERATOR========
                print('input_ids shape:', inputs.shape)
                print('inputs_raw shape:', inputs_raw.shape)
                print('perm_mask shape:', perm_mask.shape)
                print('new_targets shape:', new_targets.shape)
                print('target_mask shape:', target_mask.shape)
                print('target_mappings shape:', target_mappings.shape)
                # print('new_targets:\n', new_targets)
                # print('target_mask:\n', target_mask)
                """
                **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
                    Indices of input sequence tokens in the vocabulary.
                    Indices can be obtained using :class:`pytorch_transformers.XLNetTokenizer`.
                    See :func:`pytorch_transformers.PreTrainedTokenizer.encode` and
                    :func:`pytorch_transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.

                **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
                    Mask to avoid performing attention on padding token indices.
                    Mask values selected in ``[0, 1]``:
                    ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

                **perm_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, sequence_length)``:
                    Mask to indicate the attention pattern for each input token with values selected in ``[0, 1]``:
                    If ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
                    if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.
                    If None, each token attends to all the others (full bidirectional attention).
                    Only used during pretraining (to define factorization order) or for sequential decoding (generation).
                
                **target_mapping**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, num_predict, sequence_length)``:
                    Mask to indicate the output tokens to use.
                    If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
                    Only used during pretraining for partial prediction or for sequential decoding (generation).
                """
                # print("inputs_raw:", inputs_raw)
                # inputs_raw.double()
                inputs_raw = inputs_raw.float()
                # print("inputs_raw:", inputs_raw)
                print("inputs_raw type:", inputs_raw.type())
                print("inputs type:", inputs.type())
                print("perm_mask type:", perm_mask.type())
                print("target_mappings type:", target_mappings.type()) 
            
                outputs = model(inputs, inputs_raw, None, None, None, None, perm_mask, target_mappings)
                print("outputs[0][0]", outputs[0][0])
                print("outputs[1]", outputs[1])
                print("outputs len", len(outputs))
                print("outputs[0] shape:", outputs[0].shape)
                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

        # Save a trained model
        if args.do_train and ( n_gpu > 1 and torch.distributed.get_rank() == 0  or n_gpu <=1):
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model.save_pretrained(args.output_dir)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


if __name__ == "__main__":
    main()
