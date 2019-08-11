#!/bin/bash
if [[ $# -ne 1 ]]; then
  echo "train.sh <batch_size>"
  exit 1
fi
prefix="mt-dnn-sst"
BATCH_SIZE=$1
tstr=$(date +"%FT%H%M")
BASE_DIR="../../../mt-dnn"

train_datasets="sst"
test_datasets="sst"
MODEL_ROOT="checkpoints"
BERT_PATH="${BASE_DIR}/mt_dnn_models/mt_dnn_base_uncased.pt"
DATA_DIR="${BASE_DIR}/data/mt_dnn_uncased_lower"

answer_opt=0
optim="adamax"
grad_clipping=0
global_grad_clipping=1
lr="2e-5"

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python3 ${BASE_DIR}/train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --learning_rate ${lr} --cuda "True"
