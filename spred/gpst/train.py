# coding=utf-8
""" OpenAI GPT model fine-tuning script. """
import os
import sys
import time
import random
import logging
import argparse
import datetime
from typing import Tuple

# Third-party imports.
import optuna
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, WarmupLinearSchedule
from transformers.configuration_openai import OpenAIGPTConfig

# XLA.
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# External module imports.
from trunk import get_log
from dataset import GPSTDataset
from arguments import get_args
from modeling_openai import ConditionalGPSTModel

DEBUG = False
# pylint: disable=invalid-name, no-member, bad-continuation


# pylint: disable=protected-access
def setup(
    args: argparse.Namespace
) -> Tuple[
    argparse.Namespace,
    ConditionalGPSTModel,
    torch.device,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler._LRScheduler,
    DataLoader,
]:
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
    model : ``ConditionalGPSTModel``.
        Loaded model, set to ``train`` mode.
    device : ``torch.device``.
        Device onto which the model is loaded.
    optimizer : ``torch.optim.Optimizer``.
        PyTorch optimizer for training.
    scheduler : ``torch.optim.lr_scheduler._LRScheduler``.
        Learning rate scheduler.
    train_dataloader : ``DataLoader``.
        PyTorch object we iterate over to get training examples.
    """

    if args is None:
        raise ValueError("Argument parsing required.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.info("device: %s, n_gpu %d", str(device), n_gpu)

    # XLA device.
    device = xm.xla_device()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Config.
    config = OpenAIGPTConfig.from_pretrained(args.gpst_model)
    assert config.n_positions == config.n_ctx
    args.seq_len = config.n_ctx
    args.dim = config.input_dim
    args.orderbook_depth = config.orderbook_depth

    # Model.
    model = ConditionalGPSTModel(config)

    # Dataset.
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

    # XLA sampler.
    train_sampler = RandomSampler(train_data)
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
        )

    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size
    )

    # Optimizer.
    num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    dec = [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_dec = [p for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {"params": dec, "weight_decay": 0.01},
        {"params": no_dec, "weight_decay": 0.0},
    ]

    # Scale learning rate to num cores.
    args.learning_rate = args.learning_rate * xm.xrt_world_size()

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler.
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=(args.warmup_proportion * num_train_optimization_steps),
        t_total=num_train_optimization_steps,
    )

    # DataParallel.
    if False:
        if torch.cuda.device_count() > 1 and False:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)

    model.to(device)

    return args, model, optimizer, scheduler, device, train_dataloader


def _xmp_fn(index: int, args: argparse.Namespace) -> None:
    """ XLA multiprocessing process function. """
    xla_train(args)


def xla_run(args) -> None:
    """ Spawn XLA processes. """

    # HARDCODE
    args.log = get_log("rain")

    # HARDCODE
    nprocs = 8
    xmp.spawn(_xmp_fn, args=(args,), nprocs=nprocs)


def xla_loop(
    args: argparse.Namespace,
    model: ConditionalGPSTModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    loader: DataLoader,
) -> float:
    """ Single epoch loop. """

    # XLA tracker.
    tracker = xm.RateTracker()

    # Local variables for shape check.
    bsz = args.train_batch_size

    nb_tr_steps, tr_loss, exp_average_loss = 0.0, 0.0, 0.0
    losses = []
    completed_first_iteration = False
    tqdm_bar = tqdm(loader, desc="Training", position=0, leave=True)
    for _, batch in enumerate(tqdm_bar):

        batch = tuple(t.to(device) for t in batch)
        input_ids, position_ids, labels, inputs_raw = batch

        # Cast data to float tensor.
        # TODO: Is this necessary? What is its type before and after?
        inputs_raw = inputs_raw.float()
        labels = labels.long()

        # Handles lack of batch size data truncation in dataset class.
        if not args.stationarize and input_ids.shape[0] < args.train_batch_size:
            continue

        # Shape check.
        assert input_ids.shape == (bsz, args.seq_len)
        assert position_ids.shape == (bsz, args.seq_len)
        assert labels.shape == (bsz, args.seq_len)
        assert inputs_raw.shape == (bsz, args.seq_len, args.dim)

        outputs = model(input_ids, position_ids, labels, inputs_raw)

        loss = outputs[0]
        loss.backward()
        loss_scalar = float(loss)

        # Logging.
        losses.append(loss_scalar)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        xm.optimizer_step(optimizer)
        optimizer.zero_grad()
        scheduler.step()
        tracker.add(bsz)

        tr_loss += loss.item()
        exp_average_loss = (
            loss.item()
            if completed_first_iteration
            else 0.7 * exp_average_loss + 0.3 * loss.item()
        )

        completed_first_iteration = True
        nb_tr_steps += 1

        # Stats.
        epoch_avg_loss = np.mean(losses)
        epoch_stddev_loss = np.std(losses)
        tqdm_bar.desc = "Epoch loss dist:: mean: {:.2e}".format(
            epoch_avg_loss
        ) + " std: {:.2e}".format(epoch_stddev_loss)

    return epoch_avg_loss


def xla_train(args: argparse.Namespace) -> float:
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

    # Get model.
    args, model, optimizer, scheduler, device, train_dataloader = setup(args)

    # Optuna early stopping.
    if "trial" in args:
        trial = args.trial

    # Save names.
    weights_name = args.model_name + ".bin"
    config_name = args.model_name + ".json"

    # Flush output before we begin training.
    sys.stdout.flush()

    # Main training loop.
    start = time.time()
    nb_tr_steps, tr_loss, exp_average_loss = 0.0, 0.0, 0.0
    model.train()
    elapsed_epochs = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):

        para_loader = pl.ParallelLoader(train_dataloader, [device])

        # Single epoch loop call.
        epoch_avg_loss = xla_loop(
            args,
            model,
            optimizer,
            scheduler,
            device,
            para_loader.per_device_loader(device),
        )

        args.log.write("Epoch loss: %f" % epoch_avg_loss)

        if "trial" in args:
            trial.report(epoch_avg_loss, time.time() - start)
            if trial.should_prune():
                raise optuna.structs.TrialPruned()

        # Save every ``args.save_freq`` epochs.
        elapsed_epochs += 1

        if elapsed_epochs % args.save_freq == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            output_model_file = os.path.join(args.output_dir, weights_name)
            output_config_file = os.path.join(args.output_dir, config_name)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)

        if args.timeout > 0 and time.time() - start >= args.timeout:
            break

    return epoch_avg_loss


def train(args: argparse.Namespace) -> float:
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

    # Get model.
    args, model, optimizer, scheduler, device, train_dataloader = setup(args)

    # Optuna early stopping.
    if "trial" in args:
        trial = args.trial

    # Save names.
    weights_name = args.model_name + ".bin"
    config_name = args.model_name + ".json"

    # Local variables for shape check.
    bsz = args.train_batch_size

    # Flush output before we begin training.
    sys.stdout.flush()

    # Main training loop.
    start = time.time()
    nb_tr_steps, tr_loss, exp_average_loss = 0.0, 0.0, 0.0
    completed_first_iteration = False
    model.train()
    elapsed_epochs = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0.0
        losses = []
        nb_tr_steps = 0
        tqdm_bar = tqdm(train_dataloader, desc="Training", position=0, leave=True)
        for _, batch in enumerate(tqdm_bar):

            batch = tuple(t.to(device) for t in batch)
            input_ids, position_ids, labels, inputs_raw = batch

            # Cast data to float tensor.
            # TODO: Is this necessary? What is its type before and after?
            inputs_raw = inputs_raw.float()
            labels = labels.long()

            # Handles lack of batch size data truncation in dataset class.
            if not args.stationarize and input_ids.shape[0] < args.train_batch_size:
                continue

            # Shape check.
            assert input_ids.shape == (bsz, args.seq_len)
            assert position_ids.shape == (bsz, args.seq_len)
            assert labels.shape == (bsz, args.seq_len)
            assert inputs_raw.shape == (bsz, args.seq_len, args.dim)

            outputs = model(input_ids, position_ids, labels, inputs_raw)

            loss = outputs[0]
            loss.backward()
            loss_scalar = float(loss)

            # Logging.
            losses.append(loss_scalar)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            tr_loss += loss.item()
            exp_average_loss = (
                loss.item()
                if completed_first_iteration
                else 0.7 * exp_average_loss + 0.3 * loss.item()
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
            model_to_save = model.module if hasattr(model, "module") else model
            output_model_file = os.path.join(args.output_dir, weights_name)
            output_config_file = os.path.join(args.output_dir, config_name)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)

        if args.timeout > 0 and time.time() - start >= args.timeout:
            break

    return epoch_avg_loss


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER = get_args(PARSER)
    ARGS = PARSER.parse_args()
    xla_run(ARGS)
