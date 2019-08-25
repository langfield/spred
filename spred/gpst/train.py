# coding=utf-8
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    Itself adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py
"""
import os
import sys
import random
import logging
import argparse
import datetime
import time

import numpy as np

from tqdm import tqdm, trange
from pytorch_transformers import AdamW, WarmupLinearSchedule

# pylint: disable=wrong-import-order
# pylint: disable=no-name-in-module
# pylint: ungrouped-imports
import torch
from torch.utils.data import DataLoader

if torch.__version__[:5] == "0.3.1":
    from torch.autograd import Variable
    from torch_addons.sampler import RandomSampler
else:
    from torch.utils.data import RandomSampler

# pylint: disable=wrong-import-position
from dataset import GPSTDataset
from arguments import get_args

from modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTConfig

DEBUG = False
LOSS = 0
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# pylint: disable=invalid-name, no-member
datestring = str(datetime.datetime.now())
datestring = datestring.replace(" ", "_")
logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler("logs/rain_" + datestring + ".log"))


def train(args=None) -> float:
    """ Train a GPST Model with the arguments parsed via ``arguments.py``.
        Should be run via ``rain.sh``.
    """
    if args is None:
        parser = argparse.ArgumentParser()
        parser = get_args(parser)
        args = parser.parse_args()

    weights_name = args.model_name + ".bin"
    config_name = args.model_name + ".json"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.__version__[:5] != "0.3.1":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.__version__[:5] != "0.3.1":
        logger.info("device: %s, n_gpu %d", str(device), n_gpu)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config = OpenAIGPTConfig.from_pretrained(args.gpst_model)
    model = OpenAIGPTLMHeadModel(config)

    if torch.__version__[:5] == "0.3.1":
        model.cuda()
    else:
        model.to(device)

    # Compute the max input length for the Transformer
    max_length = model.config.n_positions

    train_data = GPSTDataset(
        args.dataset,
        max_length,
        stationarization=args.stationarize,
        aggregation_size=args.aggregation_size,
        normalization=args.normalize,
        seq_norm=args.seq_norm,
        train_batch_size=args.train_batch_size,
    )
    print("Length of training dataset:", len(train_data))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=(args.warmup_proportion * num_train_optimization_steps),
        # warmup_steps=args.warmup_steps,
        t_total=num_train_optimization_steps,
    )

    # Main training loop.
    start = time.time()
    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
    model.train()
    elapsed_epochs = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        epoch_avg_loss = 0
        nb_tr_steps = 0
        tqdm_bar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
        for _, batch in enumerate(tqdm_bar):

            if torch.__version__[:5] == "0.3.1":
                batch = tuple(t.cuda() for t in batch)
            else:
                batch = tuple(t.to(device) for t in batch)
            input_ids, position_ids, lm_labels, inputs_raw, targets_raw = batch
            inputs_raw = inputs_raw.float()
            targets_raw = targets_raw.float()

            # Shape check.
            # ===HACK===
            # Compensates for lack of batch size data truncation in
            # ``SAMPLE`` branch of ``GPSTDatatset`` class.
            if not args.stationarize and input_ids.shape[0] < args.train_batch_size:
                continue
            # ===HACK===
            assert input_ids.shape == (args.train_batch_size, max_length)
            assert position_ids.shape == (args.train_batch_size, max_length)
            assert lm_labels.shape == (args.train_batch_size, max_length)
            assert inputs_raw.shape == (
                args.train_batch_size,
                max_length,
                model.config.vocab_size,
            )
            assert targets_raw.shape == (
                args.train_batch_size,
                max_length,
                model.config.vocab_size,
            )

            # torch_0.3.1 casting.
            if torch.__version__[:5] == "0.3.1":
                position_ids = Variable(position_ids).contiguous()
                targets_raw = Variable(targets_raw.contiguous())
                inputs_raw = Variable(inputs_raw.contiguous())

            # Get only fourth column (close).
            targets_raw = targets_raw[:, :, 3]

            if DEBUG:
                print("=======================================")
                print("Type of input_ids:", type(input_ids))
                print("Type of position_ids:", type(position_ids))
                print("Type of lm_labels:", type(lm_labels))
                print("Type of inputs_raw:", type(inputs_raw))
                print("Type of targets_raw:", type(targets_raw))
                if torch.__version__[:5] == "0.3.1":
                    print("Type of position_ids data:", type(position_ids.data))
                    print("Type of targets_raw data:", type(targets_raw.data))
                print("Shape of input_ids:", input_ids.shape)
                print("Shape of position_ids:", position_ids.shape)
                print("Shape of lm_labels:", lm_labels.shape)
                print("Shape of inputs_raw:", inputs_raw.shape)
                print("Shape of targets_raw:", targets_raw.shape)

            # Forward call.
            outputs = model(input_ids, position_ids, lm_labels, inputs_raw, targets_raw)
            loss = outputs[0]
            LOSS = float(loss)
            epoch_avg_loss += LOSS
            loss.backward()
            if torch.__version__[:5] != "0.3.1":
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            if torch.__version__[:5] == "0.3.1":
                loss_data = float(loss.data)
                tr_loss += loss_data
                exp_average_loss = (
                    loss_data
                    if exp_average_loss is None
                    else 0.7 * exp_average_loss + 0.3 * loss_data
                )
            else:
                tr_loss += loss.item()
                exp_average_loss = (
                    loss.item()
                    if exp_average_loss is None
                    else 0.7 * exp_average_loss + 0.3 * loss.item()
                )

            nb_tr_steps += 1
            tqdm_bar.desc = "Training loss: {:.2e}".format(exp_average_loss)

        epoch_avg_loss = epoch_avg_loss / nb_tr_steps

        # Save every ``args.save_freq`` epochs.
        elapsed_epochs += 1
        sys.stdout.flush()
        if elapsed_epochs % args.save_freq == 0:
            if DEBUG:
                print("")
                print("Saving model to:", weights_name)
                print("")
                sys.stdout.flush()
            # Only save the model itself.
            model_to_save = model.module if hasattr(model, "module") else model

            # If we save using the predefined names, we can load using ``from_pretrained``.
            output_model_file = os.path.join(args.output_dir, weights_name)
            output_config_file = os.path.join(args.output_dir, config_name)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)

        if args.timeout > 0 and time.time() - start >= args.timeout:
            break

    return epoch_avg_loss


if __name__ == "__main__":
    train()
