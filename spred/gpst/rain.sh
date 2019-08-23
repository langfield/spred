#!/bin/bash

DATASET="../exchange/concatenated_price_data/ETHUSDT_drop.csv"
GPST_MODEL="config.json"
MODEL_NAME="pytorch_model"
OUTPUT_DIR="checkpoints"

# Hyperparameters:
SEED="42"
TRAIN_BATCH_SIZE="256"
NUM_TRAIN_EPOCHS="1000"
SAVE_FREQ="100"
MAX_GRAD_NORM="3"
LEARNING_RATE="0.00035122230205223906"
WARMUP_PROPORTION="0.26"
WEIGHT_DECAY="0.00636"
ADAM_EPSILON="7.43286922292212e-08"
AGGREGATION_SIZE="15"
STATIONARIZE="--stationarize"
NORMALIZE="--normalize"

if [ "$(whoami)" != "mckade" ]; then
    srun -J gpst -w adamantium --mem 10000 -c 4 python3 train.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME}  --seed ${SEED} --train_batch_size ${TRAIN_BATCH_SIZE} --num_train_epochs ${NUM_TRAIN_EPOCHS} --save_freq ${SAVE_FREQ} --max_grad_norm ${MAX_GRAD_NORM} --learning_rate ${LEARNING_RATE} --warmup_proportion ${WARMUP_PROPORTION} --weight_decay ${WEIGHT_DECAY} --adam_epsilon ${ADAM_EPSILON} --aggregation_size ${AGGREGATION_SIZE} ${STATIONARIZE} ${NORMALIZE}
else
    NORMALIZE=""
    DATASET="../../../ETHUSDT_drop.csv"
    AGGREGATION_SIZE="30"
    NUM_TRAIN_EPOCHS="10000"
    SAVE_FREQ="100"
    python3 train.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME}  --seed ${SEED} --train_batch_size ${TRAIN_BATCH_SIZE} --num_train_epochs ${NUM_TRAIN_EPOCHS} --save_freq ${SAVE_FREQ} --max_grad_norm ${MAX_GRAD_NORM} --learning_rate ${LEARNING_RATE} --warmup_proportion ${WARMUP_PROPORTION} --weight_decay ${WEIGHT_DECAY} --adam_epsilon ${ADAM_EPSILON} --aggregation_size ${AGGREGATION_SIZE} ${STATIONARIZE} ${NORMALIZE}
fi

