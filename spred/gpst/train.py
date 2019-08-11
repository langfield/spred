# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name openai-gpt \
          --do_train \
          --do_eval \
          --train_dataset $ROC_STORIES_DIR/cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv \
          --eval_dataset $ROC_STORIES_DIR/cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv \
          --output_dir ../log \
          --train_batch_size 16 \
"""
import argparse
import os
import csv
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader

if torch.__version__[:5] == "0.3.1":
    from torch.autograd import Variable
    from torch_addons.sampler import RandomSampler
else:
    from torch.utils.data import RandomSampler

from pytorch_transformers import (AdamW, WarmupLinearSchedule, cached_path, WEIGHTS_NAME,
                                  CONFIG_NAME)
from dataset import GPSTDataset
from modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTConfig

DEBUG = False

ROCSTORIES_URL = "https://s3.amazonaws.com/datasets.huggingface.co/ROCStories.tar.gz"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--server_ip',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")

    # Added.
    parser.add_argument("--gpst_model",
                        default=None,
                        type=str,
                        required=True,
                        help="OpenAIGPT pre-trained model path")
    args = parser.parse_args()
    print(args)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.__version__[:5] == "0.3.1":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if not torch.__version__[:5] == "0.3.1":
        logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config = OpenAIGPTConfig.from_pretrained(args.gptspred_model)
    model = OpenAIGPTLMHeadModel(config)

    if torch.__version__[:5] == "0.3.1":
        model.cuda()
    else:
        model.to(device)

    # Compute the max input length for the Transformer
    max_length = model.config.n_positions

    train_data = GPSTDataset(args.train_dataset, max_length)
    print("Length of training dataset:", len(train_data))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
            0.01
        }, {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]
        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
                                                                                      #warmup=args.warmup_proportion,
                                                                                      #max_grad_norm=args.max_grad_norm,
            weight_decay=args.weight_decay)
        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=(args.warmup_proportion *
                                                       num_train_optimization_steps),
                                         t_total=num_train_optimization_steps)

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                if torch.__version__[:5] == "0.3.1":
                    batch = tuple(t.cuda() for t in batch)
                else:
                    batch = tuple(t.to(device) for t in batch)
                input_ids, position_ids, lm_labels, inputs_raw, targets_raw = batch
                inputs_raw = inputs_raw.float()
                targets_raw = targets_raw.float()

                # Shape check.
                assert input_ids.shape == (args.train_batch_size, max_length)
                assert position_ids.shape == (args.train_batch_size, max_length)
                assert lm_labels.shape == (args.train_batch_size, max_length)
                assert inputs_raw.shape == (args.train_batch_size, max_length, inputs_raw.shape[2])

                # torch_0.3.1 casting.
                if torch.__version__[:5] == "0.3.1":
                    position_ids = Variable(position_ids).contiguous()
                    targets_raw = Variable(targets_raw.contiguous())

                if DEBUG:
                    print("=======================================")
                    print("Type of input_ids:", type(input_ids))
                    print("Type of position_ids:", type(position_ids))
                    print("Type of lm_labels:", type(lm_labels))
                    print("Type of inputs_raw:", type(inputs_raw))
                    print("Type of targets_raw:", type(targets_raw))
                    if torch.__version__[:5] == "0.3.1":
                        print("type of position_ids data:", type(position_ids.data))
                        print("Type of targets_raw data:", type(targets_raw.data))

                # Forward call.
                outputs = model(input_ids, position_ids, None, lm_labels, inputs_raw, targets_raw)
                loss = outputs[0]
                loss.backward()
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()

                if torch.__version__[:5] == "0.3.1":
                    loss_data = float(loss.data)
                    tr_loss += loss_data
                    exp_average_loss = loss_data if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss_data
                else:
                    tr_loss += loss.item()
                    exp_average_loss = loss.item(
                    ) if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()

                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e}".format(exp_average_loss)

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model,
                                                'module') else model   # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

        # Load a trained model.
        if torch.__version__[:5] == "0.3.1":
            loaded_config = OpenAIGPTConfig.from_json_file(output_config_file)
            model = OpenAIGPTLMHeadModel(loaded_config)
            model.load_state_dict(torch.load(output_model_file))
            model.cuda()
        else:
            model = OpenAIGPTLMHeadModel.from_pretrained(args.output_dir)
            model.to(device)

if __name__ == '__main__':
    main()
