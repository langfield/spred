import argparse
import optuna
import tempfile
import shutil
import json
import time
import os

from train import train, train_args


def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)


def objective(trial: optuna.Trial) -> float:
    parser = argparse.ArgumentParser()
    parser = train_args(parser)
    args = parser.parse_args()

    # Must cast args which are supposed to be ints to ints.
    args.seed = trial.suggest_int("seed", 40, 43)
    args.train_batch_size = int(trial.suggest_discrete_uniform(
        "train_batch_size", 512, 1024, 256
    ))
    args.max_grad_norm = trial.suggest_int("max_grad_norm", 1, 5)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 2.5e-4, 5e-2)
    args.warmup_steps = int(trial.suggest_discrete_uniform(
        "warmup_steps", 10000, 60000, 10000
    ))
    args.weight_decay = trial.suggest_loguniform("weight_decay", 1e-3, 3e-2)
    args.adam_epsilon = trial.suggest_loguniform("adam_epsilon", 1e-8, 1e-7)

    config = {}
    config["initializer_range"] = trial.suggest_uniform("initializer_range", 0.02, 0.05)
    config["layer_norm_epsilon"] = trial.suggest_loguniform(
        "layer_norm_epsilon", 1e-12, 5e-5
    )
    config["n_ctx"] = int(trial.suggest_discrete_uniform("n_ctx", 30, 180, 10))
    config["n_positions"] = config["n_ctx"]
    config["resid_pdrop"] = trial.suggest_uniform("resid_pdrop", 0.05, 0.15)
    config["attn_pdrop"] = trial.suggest_uniform("attn_pdrop", 0.05, 0.15)
    config["n_embd"] = 5
    config["n_head"] = 5
    config["vocab_size"] = 1  # Dummy value.

    dirpath = tempfile.mkdtemp()

    config_filename = str(time.time()) + ".json"
    config_filepath = os.path.join(dirpath, config_filename)
    with open(config_filepath, "w") as fp:
        json.dump(config, fp)

    loss = train(config_filepath, args)

    shutil.rmtree(dirpath)
    return loss


if __name__ == "__main__":
    main()