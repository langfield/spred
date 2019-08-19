#!/bin/bash

MODEL_ROOT="checkpoints"
GPST_MODEL="config.json"
TRAIN_DATASET="../exchange/concatenated_price_data/ETHUSDT_drop.csv"

if [ "$(whoami)" != "mckade" ]; then
    srun -J gpst -w vibranium --mem 10000 -c 4 python3 train.py --train_dataset ${TRAIN_DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size 2048 --num_train_epochs 10 --save_freq 50
else
    #python3 train.py --train_dataset ../../../ETHUSDT_ta_drop.csv --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size 512 --num_train_epochs 10000 --save_freq 20
    python3 optimize.py --train_dataset ../../../ETHUSDT_drop.csv --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size 1024 --num_train_epochs 1500 --save_freq 20 --timeout 30
fi

