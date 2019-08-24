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
    args.stationarize = True
    args.normalize = True
    args.seed = 42
    args.max_grad_norm = 3
    args.warmup_steps = 10000
    args.weight_decay = 0.005
    args.adam_epsilon = 7.400879524874149e-08
    args.learning_rate = trial.suggest_loguniform("learning_rate", 8e-7, 5e-4)
    args.warmup_proportion = trial.suggest_uniform("warmup_proportion", 0.05, 0.4)
    batch_size = trial.suggest_discrete_uniform("train_batch_size", 32, 256, 32)
    args.train_batch_size = int(batch_size)
    agg_size = 1
    args.aggregation_size = int(agg_size)

    # Set config.
    config = {}
    config["initializer_range"] = 0.039589260915990014
    config["layer_norm_epsilon"] = 9.064249722000914e-11
    config["n_ctx"] = int(trial.suggest_discrete_uniform("n_ctx", 5, 30, 5))
    config["n_positions"] = config["n_ctx"]
    config["resid_pdrop"] = 0.08
    config["attn_pdrop"] = 0.08
    config["n_embd"] = int(trial.suggest_discrete_uniform("n_embd", 32, 768, 64))
    config["n_head"] = 16
    config["vocab_size"] = 5
    config["n_layer"] = 12

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
