# coding=utf-8
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    Itself adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py
"""
import os
import sys
import time
import copy
import random
import logging
import argparse
import datetime
from typing import Tuple, Dict, List

# Third-party imports.
import optuna
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, RandomSampler

# External module imports.
from arguments import get_args
from dataset import GPSTDataset
from modeling_openai import OpenAIGPTLMHeadModel, OpenAIGPTConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule

DEBUG = False
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


# pylint: disable=protected-access
def setup(
    args: argparse.Namespace = None
) -> Tuple[argparse.Namespace, OpenAIGPTConfig, torch.device, DataLoader]:
    """
    Training model, dataset, and optimizer setup.

    Parameters
    ----------
    args : ``args.Namespace``.
        Training arguments.

    Returns
    -------
    args : ``args.Namespace``.
        Updated training arguments.
    model : ``OpenAIGPTLMHeadModel``.
        Loaded model, set to ``train`` mode.
    optimizer : ``torch.optim.Optimizer``.
        PyTorch optimizer for training.
    scheduler : ``torch.optim.lr_scheduler._LRScheduler``.
        Learning rate scheduler.
    train_dataloader : ``DataLoader``.
        PyTorch object we iterate over to get training examples.
    """

    if args is None:
        parser = argparse.ArgumentParser()
        parser = get_args(parser)
        args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info("device: %s, n_gpu %d", str(device), n_gpu)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ===MOD===
    global_config = OpenAIGPTConfig.from_pretrained(args.gpst_model)
    assert global_config.n_positions == global_config.n_ctx
    args.seq_len = global_config.n_ctx
    args.dim = global_config.input_dim
    args.orderbook_depth = global_config.orderbook_depth

    train_data = GPSTDataset(
        args.dataset,
        args.seq_len,
        args.dim,
        args.orderbook_depth,
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
    # ===MOD===

    return args, global_config, device, train_dataloader


def spin_models(
    args: argparse.Namespace,
    global_config: OpenAIGPTConfig,
    device: torch.device,
    train_dataloader: DataLoader,
) -> Tuple[torch.nn.ModuleDict, Dict[str, AdamW], Dict[str, WarmupLinearSchedule]]:
    """ Spins up models for different modes. """

    model_dict: torch.nn.ModuleDict = torch.nn.ModuleDict()
    optimizer_dict: Dict[str, AdamW] = {}
    scheduler_dict: Dict[str, WarmupLinearSchedule] = {}

    for mode in global_config.modes:
        config = copy.deepcopy(global_config)
        config.mode = mode
        model = OpenAIGPTLMHeadModel(config)

        # Prepare optimizer.
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        dec = [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
        no_dec = [p for n, p in param_optimizer if any(nd in n for nd in no_decay)]
        optimizer_grouped_parameters = [
            {"params": dec, "weight_decay": 0.01},
            {"params": no_dec, "weight_decay": 0.0},
        ]

        # --------Optimizer--------
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
        )
        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=(args.warmup_proportion * num_train_optimization_steps),
            t_total=num_train_optimization_steps,
        )
        # --------Optimizer--------

        if torch.cuda.device_count() > 1 and False:
            model_dict[mode] = torch.nn.DataParallel(model)
        else:
            model_dict[mode] = model
        optimizer_dict[mode] = optimizer
        scheduler_dict[mode] = scheduler

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    model_dict.to(device)

    return model_dict, optimizer_dict, scheduler_dict


def train(args: argparse.Namespace = None) -> float:
    """
    Train a GPST Model with the arguments parsed via ``arguments.py``.
    Should be run via ``rain.sh``.

    Parameters
    ----------
    args : ``argparse.Namespace``.
        Training arguments with which we load dataset, create model.

    Returns
    -------
    epoch_avg_loss : ``float``.
        The average loss for the last epoch.
    """
    # args, model, optimizer, scheduler, train_dataloader = setup(args)
    args, global_config, device, train_dataloader = setup(args)
    model_dict, optimizer_dict, scheduler_dict = spin_models(
        args, global_config, device, train_dataloader
    )

    # Optuna early stopping.
    if "trial" in args:
        trial = args.trial

    # Save names.
    weights_names: Dict[str, str] = {}
    config_names: Dict[str, str] = {}
    for mode, _ in model_dict.items():
        weights_names[mode] = "%s_%s.bin" % (args.model_name, mode)
        config_names[mode] = "%s_%s.json" % (args.model_name, mode)

    # Local variables for shape check.
    bsz = args.train_batch_size
    depth_range = 2 * args.orderbook_depth + 1

    # Flush output before we begin training.
    sys.stdout.flush()

    # Main training loop.
    start = time.time()
    nb_tr_steps, tr_loss, exp_average_loss = 0.0, 0.0, 0.0
    completed_first_iteration = False
    model_dict.train()
    elapsed_epochs = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0.0
        losses = []
        nb_tr_steps = 0
        tqdm_bar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
        for _, batch in enumerate(tqdm_bar):

            batch = tuple(t.to(device) for t in batch)

            input_ids = batch[0]
            position_ids = batch[1]
            bid_classif_labels = batch[2]
            bid_increase_labels = batch[3]
            bid_decrease_labels = batch[4]
            ask_classif_labels = batch[5]
            ask_increase_labels = batch[6]
            ask_decrease_labels = batch[7]
            inputs_raw = batch[8]
            inputs_raw = inputs_raw.float()

            labels_dict: Dict[str, torch.Tensor] = {}
            labels_dict["bid_classification"] = bid_classif_labels
            labels_dict["bid_increase"] = bid_increase_labels
            labels_dict["bid_decrease"] = bid_decrease_labels
            labels_dict["ask_classification"] = ask_classif_labels
            labels_dict["ask_increase"] = ask_increase_labels
            labels_dict["ask_decrease"] = ask_decrease_labels

            # Handles lack of batch size data truncation in dataset class.
            if not args.stationarize and input_ids.shape[0] < args.train_batch_size:
                continue

            # Shape check.
            assert input_ids.shape == (bsz, args.seq_len)
            assert position_ids.shape == (bsz, args.seq_len)
            # ===MOD===
            # ===MOD===
            assert inputs_raw.shape == (bsz, args.seq_len, args.dim)

            # Forward calls.
            model_losses: List[float] = []
            for mode, model in model_dict.items():

                labels = labels_dict[mode].long()
                if mode[:3] == "bid":
                    assert labels.shape == (bsz, args.seq_len)
                elif mode[:3] == "ask":
                    assert labels.shape == (bsz, args.seq_len, depth_range)
                else:
                    # TODO: fix error message.
                    raise ValueError("Mode has invalid value")

                outputs = model(input_ids, position_ids, labels, inputs_raw)

                loss = outputs[0]
                loss.backward()
                loss_scalar = float(loss)
                model_losses.append(loss_scalar)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer_dict[mode].step()
                optimizer_dict[mode].zero_grad()
                scheduler_dict[mode].step()

            # Logging.
            sum_loss = sum(model_losses)
            losses.append(sum_loss)

            tr_loss += sum_loss
            exp_average_loss = (
                sum_loss
                if completed_first_iteration
                else 0.7 * exp_average_loss + 0.3 * sum_loss
            )

            completed_first_iteration = True
            nb_tr_steps += 1

            # Stats.
            epoch_avg_loss = np.mean(losses)
            epoch_stddev_loss = np.std(losses)
            tqdm_bar.desc = "Epoch loss dist:: mean: {:.2e}".format(
                epoch_avg_loss
            ) + " std: {:.2e}".format(epoch_stddev_loss)

        if "trial" in args:
            trial.report(epoch_avg_loss, time.time() - start)
            if trial.should_prune():
                raise optuna.structs.TrialPruned()

        # Save every ``args.save_freq`` epochs.
        elapsed_epochs += 1
        if elapsed_epochs % args.save_freq == 0:
            for mode, model in model_dict.items():
                model_to_save = model.module if hasattr(model, "module") else model
                output_model_file = os.path.join(args.output_dir, weights_names[mode])
                output_config_file = os.path.join(args.output_dir, config_names[mode])

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)

        if args.timeout > 0 and time.time() - start >= args.timeout:
            break

    return epoch_avg_loss


if __name__ == "__main__":
    train()
