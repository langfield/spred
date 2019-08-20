#!/bin/bash

MODEL_ROOT="checkpoints"
GPST_MODEL="config.json"
TRAIN_DATASET="../exchange/concatenated_price_data/ETHUSDT_drop.csv"

if [ "$(whoami)" != "mckade" ]; then
    # srun -J gpst -w vibranium --mem 10000 -c 4 python3 train.py --train_dataset ${TRAIN_DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size 128 --num_train_epochs 1000 --save_freq 1
    srun -J gpst -w adamantium --mem 10000 -c 4 python3 train.py --train_dataset ${TRAIN_DATASET} --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --seed 42 --train_batch_size 512 --num_train_epochs 1000 --save_freq 1 --max_grad_norm 3 --learning_rate 4.1611368011568754e-02 --warmup_proportion 0.31677 --weight_decay 0.00636 --adam_epsilon 1.03286922292212e-08 --no_price_preprocess
else
    #python3 train.py --train_dataset ../../../ETHUSDT_ta_drop.csv --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size 512 --num_train_epochs 10000 --save_freq 20
    python3 train.py --train_dataset ../../../ETHUSDT_drop.csv --gpst_model ${GPST_MODEL} --output_dir ${MODEL_ROOT} --do_train --train_batch_size 512 --num_train_epochs 1000 --save_freq 20 --no_price_preprocess #--timeout 30
fi

