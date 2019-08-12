#!/bin/bash
prefix="mt-dnn-sst"
tstr=$(date +"%FT%H%M")

test_datasets="sst"
MODEL_ROOT="checkpoints"
# Requires absolute path.
# BERT_PATH="/home/mckade/spred/mt-dnn/scripts/checkpoints/sst_finetune/model_0.pt"
#BERT_PATH="/homes/3/whitaker.213/packages/data/mt-dnn/mt_dnn_models/model_0.pt"
BERT_PATH="checkpoints/sst_finetune/model_0.pt"
DATA_DIR="data/mt_dnn_uncased"

model_dir="checkpoints/${prefix}_eval_${tstr}"
log_file="${model_dir}/log.log"
python3 eval.py --init_checkpoint ${BERT_PATH} --output_dir ${model_dir} --log_file ${log_file} --cuda "True"
