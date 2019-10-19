#!/bin/bash

DATASET="/root/books/sampleset.csv"
SEP=","
GPST_MODEL="config.json"
MODEL_NAME="orderbook"
OUTPUT_DIR="ckpts"

# Hyperparameters.
SEED="42"
EVAL_BATCH_SIZE="1"
AGGREGATION_SIZE="1"
GRAPH_DIR="graphs/"
WIDTH="2000"
TERMINAL_PLOT_WIDTH="50"

# Format: ``--<argument_name>``.
NORMALIZE=""
STATIONARIZE=""
SEQ_NORM=""

if [ "$(whoami)" != "mckade" ]; then
    python3 eval.py --gpst_model ${GPST_MODEL} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --eval_batch_size ${EVAL_BATCH_SIZE} --width ${WIDTH} --dataset ${DATASET} --sep ${SEP} --graph_dir ${GRAPH_DIR} --terminal_plot_width ${TERMINAL_PLOT_WIDTH} ${NORMALIZE} ${STATIONARIZE} --aggregation_size ${AGGREGATION_SIZE} ${SEQ_NORM}
else
    DATASET="../exchange/gemini.csv"
    AGGREGATION_SIZE="1"
    STATIONARIZE="--stationarize"
    NORMALIZE=""
    python3 eval.py --gpst_model ${GPST_MODEL} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME} --eval_batch_size ${EVAL_BATCH_SIZE} --width ${WIDTH} --dataset ${DATASET} --sep ${SEP} --graph_dir ${GRAPH_DIR} --terminal_plot_width ${TERMINAL_PLOT_WIDTH} ${NORMALIZE} ${STATIONARIZE} --aggregation_size ${AGGREGATION_SIZE}
fi

