#!/bin/bash

MODEL_ROOT="ckpts"
GPST_MODEL="config.json"
DATASET="/root/books/sampleset.csv"
TIMEOUT="3600"
SAVE_FREQ="20"

python3 optimize.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --save_freq ${SAVE_FREQ} --timeout ${TIMEOUT}
