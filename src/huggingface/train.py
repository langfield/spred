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

import logging
import os
import sys
import random
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME

from args import parse_args
from perm_generator import _local_perm
from xlspred_dataset import XLSpredDataset
from xlspred_utils import prepare_optimizer, prepare_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



DEBUG = False

def main():
    
    # Get training arguments. 
    args = parse_args()

    # Prepare device.
    #===========DEVICE============
    #===========vvvvvv============
    
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

    #===========^^^^^^============
    #===========DEVICE============
    
    # Load dataset. 
    #===========DATASET===========
    #===========vvvvvvv===========

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
    
    #===========^^^^^^^===========
    #===========DATASET===========

    # Prepare model
    model = prepare_model(args, device, n_gpu)

    # Prepare optimizer, create scheduler. 
    optimizer, scheduler = prepare_optimizer(args, model, num_train_optimization_steps)

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
                inputs, inputs_raw, targets_raw, is_maskeds, targets, seg_ids, labels = batch
                
                #=======PERM GENERATOR========
                #===========vvvvvv============
                
                perm_mask = []
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

                    perm_mask_0 = torch.cat([perm_mask_0, torch.ones([reuse_len, non_reuse_len]).to(device)], 
                                            dim=1)
                    
                    perm_mask_1 = torch.cat([torch.zeros([non_reuse_len, reuse_len]).to(device), perm_mask_1],
                                            dim=1)
                    
                    perm_mask.append(torch.cat([perm_mask_0, perm_mask_1], dim=0))
                    target_mask_row = torch.cat([target_mask_0, target_mask_1], dim=0)
                    # TODO: we are currently excluding input_k and input_q

                    indices = torch.arange(0, seq_len)
                    bool_target_mask = target_mask_row.byte()
                    # Has length equal to num `True` vals in `bool_target_mask` : <= seq_len
                    indices = indices[bool_target_mask] 
                    # length of indices after removing the masked out tokens
                    index_len = indices.shape[0]

                    # extra padding due to CLS/SEP introduced after prepro
                    actual_num_predict = indices.shape[0]
                    # TODO: By how much are we supposed to pad here vvvv?
                    pad_len = seq_len - actual_num_predict
                    # pad_len = args.num_predict - actual_num_predict

                    # target mapping
                    inp = indices % seq_len
                    inp_ = torch.unsqueeze(inp, 1)
                    target_mapping = torch.FloatTensor(index_len, seq_len).zero_()
                    target_mapping.scatter_(1, inp_, 1) # Shape: (actual_num_predict, seq_len)
                    paddings = torch.zeros([pad_len, seq_len], dtype=target_mapping.dtype)
                    # print("target_mapping_row shape:", target_mapping.shape)
                    target_mapping = torch.cat([target_mapping, paddings])
                    target_mappings.append(target_mapping)

                perm_mask = torch.stack(perm_mask)

                # Shape: (bsz, actual_num_predict, seq_len)
                target_mappings = torch.stack(target_mappings)

                #===========^^^^^=============
                #=======PERM GENERATOR========
 
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
                # Cast to float so PyTorch is less angry. 
                inputs_raw = inputs_raw.float()
                targets_raw = targets_raw.float()
     
                outputs = model(inputs, inputs_raw, None, None, None, None, perm_mask, target_mappings.to(device), targets_raw)
                print("train.py: outputs[0]:", outputs[0]) 
                sys.stdout.flush()

                if DEBUG:
                    print("train.py: Loss/outputs[0]:", outputs[0]) 
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
                nb_tr_examples += inputs.size(0)
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
