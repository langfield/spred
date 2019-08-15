#!/bin/bash

MODEL_ROOT="checkpoints"
GPST_MODEL="config.json"

if [ "$(whoami)" != "mckade" ]; then
    srun -J gpst -w vibranium --mem 10000 -c 4 python3 train.py --train_dataset ../exchange/ETHUSDT_small_drop.csv --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size 8 --num_train_epochs 10 --save_freq 20
else
    python3 train.py --train_dataset ../../../ETHUSDT_drop.csv --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size 2048 --num_train_epochs 1500 --save_freq 20
fi

