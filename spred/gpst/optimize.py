""" Script for optimizing GPST model hyperparameters via Optuna. """

import os
import time
import json
import shutil
import logging
import argparse
import tempfile
import datetime

import optuna

from train import train
from arguments import get_args


def main():
    """ Run an Optuna study. """
    datestring = str(datetime.datetime.now())
    datestring = datestring.replace(" ", "_")
    logging.getLogger().setLevel(logging.INFO)  # Setup the root logger.
    logging.getLogger().addHandler(logging.FileHandler("optuna_" + datestring + ".log"))
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in stderr.

    study = optuna.create_study()
    logging.getLogger().info("Start optimization.")
    study.optimize(objective, n_trials=100)


def objective(trial: optuna.Trial) -> float:
    """ Optuna objective function. """
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()

    # Set arguments.
    args.num_train_epochs = 100000
    args.stationarize = False
    args.normalize = True
    args.aggregation_size = 1
    args.seed = 42
    args.max_grad_norm = 3
    args.warmup_steps = 10000
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e-4)
    args.warmup_proportion = trial.suggest_uniform("warmup_proportion", 0.05, 0.4)
    args.weight_decay = trial.suggest_loguniform("weight_decay", 5e-4, 1e-2)
    args.adam_epsilon = trial.suggest_loguniform("adam_epsilon", 1e-9, 1e-7)
    batch_size = trial.suggest_discrete_uniform("train_batch_size", 64, 512, 64)
    args.train_batch_size = int(batch_size)

    # Set config.
    config = {}
    config["initializer_range"] = trial.suggest_uniform("initializer_range", 0.01, 0.05)
    config["layer_norm_epsilon"] = trial.suggest_loguniform("lay_norm_eps", 1e-12, 5e-5)
    config["n_ctx"] = int(trial.suggest_discrete_uniform("n_ctx", 5, 60, 5))
    config["n_positions"] = config["n_ctx"]
    config["resid_pdrop"] = trial.suggest_uniform("resid_pdrop", 0.02, 0.15)
    config["attn_pdrop"] = trial.suggest_uniform("attn_pdrop", 0.02, 0.15)
    config["n_embd"] = int(trial.suggest_discrete_uniform("n_embd", 32, 256, 16))
    config["n_head"] = 16
    config["vocab_size"] = 5

    dirpath = tempfile.mkdtemp()
    config_filename = str(time.time()) + ".json"
    config_filepath = os.path.join(dirpath, config_filename)
    with open(config_filepath, "w") as path:
        json.dump(config, path)
    args.gpst_model = config_filepath
    args.model_name = "optuna"

    loss = train(args)

    shutil.rmtree(dirpath)
    return loss


if __name__ == "__main__":
    main()
