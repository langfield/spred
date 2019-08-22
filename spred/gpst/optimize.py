import os
import time
import json
import shutil
import argparse
import tempfile

import optuna

from train import train
from arguments import get_args


def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)


def objective(trial: optuna.Trial) -> float:
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()

    # Hyperparams to manually set.
    args.num_train_epochs = 100000
    args.stationarize = False

    # Must cast args which are supposed to be ints to ints.
    # args.seed = trial.suggest_int("seed", 40, 43)
    args.seed = 42
    args.train_batch_size = int(
        trial.suggest_discrete_uniform("train_batch_size", 16, 64, 8)
    )

    # args.max_grad_norm = trial.suggest_int("max_grad_norm", 1, 5)
    args.max_grad_norm = 3
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e-4)
    args.warmup_steps = 10000
    args.warmup_proportion = trial.suggest_uniform("warmup_proportion", 0.05, 0.4)
    args.weight_decay = trial.suggest_loguniform("weight_decay", 5e-4, 1e-2)
    args.adam_epsilon = trial.suggest_loguniform("adam_epsilon", 1e-9, 1e-7)

    config = {}
    config["initializer_range"] = trial.suggest_uniform("initializer_range", 0.01, 0.05)
    config["layer_norm_epsilon"] = trial.suggest_loguniform(
        "layer_norm_epsilon", 1e-12, 5e-5
    )
    config["n_ctx"] = int(trial.suggest_discrete_uniform("n_ctx", 5, 60, 5))
    config["n_positions"] = config["n_ctx"]
    config["resid_pdrop"] = trial.suggest_uniform("resid_pdrop", 0.02, 0.15)
    config["attn_pdrop"] = trial.suggest_uniform("attn_pdrop", 0.02, 0.15)
    # config["n_embd"] = int(trial.suggest_discrete_uniform("n_embd", 64, 256, 32))
    config["n_embd"] = 768
    config["n_head"] = 16
    config["vocab_size"] = 5  # Input data dim.

    dirpath = tempfile.mkdtemp()

    config_filename = str(time.time()) + ".json"
    config_filepath = os.path.join(dirpath, config_filename)
    with open(config_filepath, "w") as fp:
        json.dump(config, fp)
    args.gpst_model = config_filepath

    loss = train(args)

    shutil.rmtree(dirpath)
    return loss


if __name__ == "__main__":
    main()
