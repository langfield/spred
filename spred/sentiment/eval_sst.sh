#!/bin/bash
prefix="mt-dnn-sst"
tstr=$(date +"%FT%H%M")

test_datasets="sst"
BERT_PATH="checkpoints/sst_finetune/model_0.pt"
DATA_DIR="data/mt_dnn_uncased"

answer_opt=0
grad_clipping=0
global_grad_clipping=1
lr="2e-5"

model_dir="checkpoints/${prefix}_eval_${tstr}"
log_file="${model_dir}/log.log"
python3 eval.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --test_datasets ${test_datasets} --cuda "True"