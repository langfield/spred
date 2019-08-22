#!/bin/bash

TRAIN_DATASET="../exchange/concatenated_price_data/ETHUSDT_drop.csv"
GPST_MODEL="config.json"
MODEL_ROOT="checkpoints"
WEIGHTS_NAME="pytorch_model.bin"
CONFIG_NAME="config.json"

# Hyperparameters:
SEED="42"
TRAIN_BATCH_SIZE="512"
NUM_TRAIN_EPOCHS="1000"
SAVE_FREQ="1"
MAX_GRAD_NORM="3"
LEARNING_RATE="4.1611368011568754e-02"
WARMUP_PROPORTION="0.31677"
WEIGHT_DECAY="0.00636"
ADAM_EPSILON="1.03286922292212e-08"
NO_PRICE_PREPROCESS="--no_price_preprocess"
NORMALIZE="--normalize"

if [ "$(whoami)" != "mckade" ]; then
    # srun -J gpst -w vibranium --mem 10000 -c 4 python3 train.py --train_dataset ${TRAIN_DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size 128 --num_train_epochs 1000 --save_freq 1
    srun -J gpst -w adamantium --mem 10000 -c 4 python3 train.py --train_dataset ${TRAIN_DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --weights_name ${WEIGHTS_NAME} --config_name ${CONFIG_NAME} --do_train --seed ${SEED} --train_batch_size ${BATCH_SIZE} --num_train_epochs ${NUM_TRAIN_EPOCHS} --save_freq ${SAVE_FREQ} --max_grad_norm ${MAX_GRAD_NORM} --learning_rate ${LEARNING_RATE} --warmup_proportion ${WARMUP_PROPORTION} --weight_decay ${WEIGHT_DECAY} --adam_epsilon ${ADAM_EPSILON} ${NO_PRICE_PREPROCESS} ${NORMALIZE}
else
    #python3 train.py --train_dataset ../../../ETHUSDT_ta_drop.csv --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size 512 --num_train_epochs 10000 --save_freq 20
    python3 train.py --train_dataset ../../../ETHUSDT_drop.csv --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size ${TRAIN_BATCH_SIZE}  --num_train_epochs ${NUM_TRAIN_EPOCHS} --save_freq ${SAVE_FREQ} #--timeout 30
fi

