#!/bin/bash

MODEL_ROOT="checkpoints"
GPST_MODEL="config.json"
DATASET="../exchange/concatenated_price_data/ETHUSDT_drop.csv"

TIMEOUT="600"
SAVE_FREQ="1"

if [ "$(whoami)" != "mckade" ]; then
    srun -J gpst -w vibranium --mem 10000 -c 4 python3 optimize.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --save_freq ${SAVE_FREQ} --timeout ${TIMEOUT}
else
    python3 optimize.py --dataset ../exchange/ETHUSDT_small_drop.csv --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --save_freq ${SAVE_FREQ} --timeout ${TIMEOUT}
fi

