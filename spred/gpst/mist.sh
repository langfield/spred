#!/bin/bash

DATASET="../exchange/concatenated_price_data/IOTABTC_drop.csv"
GPST_MODEL="config.json"
MODEL_NAME="pytorch_model"
OUTPUT_DIR="checkpoints"

# Hyperparameters:
SEED="42"
EVAL_BATCH_SIZE="1"
AGGREGATION_SIZE="1"
GRAPH_DIR="graphs/"
WIDTH="40"
TERMINAL_PLOT_WIDTH="50"
NORMALIZE="--normalize"
STATIONARIZE=""

if [ "$(whoami)" != "mckade" ]; then
    python3 eval.py --gpst_model ${GPST_MODEL} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --eval_batch_size ${EVAL_BATCH_SIZE} --width ${WIDTH} --dataset ${DATASET} --graph_dir ${GRAPH_DIR} --terminal_plot_width ${TERMINAL_PLOT_WIDTH} ${NORMALIZE} ${STATIONARIZE}
else
    python3 eval.py --gpst_model ${GPST_MODEL} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --eval_batch_size ${EVAL_BATCH_SIZE} --width ${WIDTH} --dataset ${DATASET} --graph_dir ${GRAPH_DIR} --terminal_plot_width ${TERMINAL_PLOT_WIDTH} ${NORMALIZE} ${STATIONARIZE}
fi

