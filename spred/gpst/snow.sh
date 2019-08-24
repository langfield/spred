#!/bin/bash

MODEL_ROOT="checkpoints"
GPST_MODEL="config.json"
DATASET="../exchange/concatenated_price_data/ETHUSDT_drop.csv"

TIMEOUT="3000"
SAVE_FREQ="1"

if [ "$(whoami)" != "mckade" ]; then
    srun -J gpst -w vibranium --mem 10000 -c 4 python3 optimize.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --save_freq ${SAVE_FREQ} --timeout ${TIMEOUT}
    # python3 optimize.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --save_freq ${SAVE_FREQ} --timeout ${TIMEOUT}
else
    python3 optimize.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --save_freq ${SAVE_FREQ} --timeout ${TIMEOUT}
fi

