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
WIDTH="300"
TERMINAL_PLOT_WIDTH="250"
NORMALIZE="--normalize"
STATIONARIZE=""

if [ "$(whoami)" != "mckade" ]; then
    python3 eval.py --gpst_model ${GPST_MODEL} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --eval_batch_size ${EVAL_BATCH_SIZE} --width ${WIDTH} --dataset ${DATASET} --graph_dir ${GRAPH_DIR} --terminal_plot_width ${TERMINAL_PLOT_WIDTH} ${NORMALIZE} ${STATIONARIZE} --aggregation_size ${AGGREGATION_SIZE}
else
    DATASET="../../../ETHUSDT_drop_eval.csv"
    AGGREGATION_SIZE="1"
    STATIONARIZE="--stationarize"
    NORMALIZE=""
    python3 eval.py --gpst_model ${GPST_MODEL} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --eval_batch_size ${EVAL_BATCH_SIZE} --width ${WIDTH} --dataset ${DATASET} --graph_dir ${GRAPH_DIR} --terminal_plot_width ${TERMINAL_PLOT_WIDTH} ${NORMALIZE} ${STATIONARIZE} --aggregation_size ${AGGREGATION_SIZE}
fi

