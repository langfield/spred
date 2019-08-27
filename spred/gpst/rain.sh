#!/bin/bash

DATASET="../exchange/concatenated_price_data/ETHUSDT_prepro.csv"
GPST_MODEL="config.json"
MODEL_NAME="optuna_best_8_27"
OUTPUT_DIR="checkpoints"

# Hyperparameters:
SEED="42"
TRAIN_BATCH_SIZE="224"
NUM_TRAIN_EPOCHS="1000"
SAVE_FREQ="10"
MAX_GRAD_NORM="3"
LEARNING_RATE="0.000433"
WARMUP_PROPORTION="0.05696"
WEIGHT_DECAY="0.005"
ADAM_EPSILON="7.400879524874149e-08"
AGGREGATION_SIZE="1"
STATIONARIZE="--stationarize"
NORMALIZE=""
SEQ_NORM="--seq_norm"

if [ "$(whoami)" != "mckade" ]; then
    srun -J gpst -w adamantium --mem 10000 -c 4 python3 train.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME}  --seed ${SEED} --train_batch_size ${TRAIN_BATCH_SIZE} --num_train_epochs ${NUM_TRAIN_EPOCHS} --save_freq ${SAVE_FREQ} --max_grad_norm ${MAX_GRAD_NORM} --learning_rate ${LEARNING_RATE} --warmup_proportion ${WARMUP_PROPORTION} --weight_decay ${WEIGHT_DECAY} --adam_epsilon ${ADAM_EPSILON} --aggregation_size ${AGGREGATION_SIZE} ${STATIONARIZE} ${NORMALIZE} ${SEQ_NORM}
else
    NORMALIZE=""
    DATASET="../../../ETHUSDT_drop.csv"
    AGGREGATION_SIZE="1"
    NUM_TRAIN_EPOCHS="1000"
    SAVE_FREQ="20"
    python3 train.py --dataset ${DATASET} --gpst_model ${GPST_MODEL} --output_dir ${OUTPUT_DIR} --model_name ${MODEL_NAME}  --seed ${SEED} --train_batch_size ${TRAIN_BATCH_SIZE} --num_train_epochs ${NUM_TRAIN_EPOCHS} --save_freq ${SAVE_FREQ} --max_grad_norm ${MAX_GRAD_NORM} --learning_rate ${LEARNING_RATE} --warmup_proportion ${WARMUP_PROPORTION} --weight_decay ${WEIGHT_DECAY} --adam_epsilon ${ADAM_EPSILON} --aggregation_size ${AGGREGATION_SIZE} ${STATIONARIZE} ${NORMALIZE}
fi

