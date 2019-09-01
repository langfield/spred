#!/bin/bash

MODEL_ROOT="ckpts"
GPST_MODEL="config.json"
DATASET="../exchange/concatenated_price_data/ETHUSDT_prepro.csv"
TIMEOUT="1800"
SAVE_FREQ="10"

if [ "$(whoami)" != "mckade" ]; then
    srun -J gpst -w vibranium --mem 10000 -c 4 python3 optimize.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --save_freq ${SAVE_FREQ} --timeout ${TIMEOUT}
else
    python3 optimize.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --save_freq ${SAVE_FREQ} --timeout ${TIMEOUT}
fi

