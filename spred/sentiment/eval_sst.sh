#!/bin/bash
PREFIX="mt-dnn-sst"
TSTR=$(date +"%FT%H%M")

MODEL_ROOT="checkpoints"
BERT_PATH="checkpoints/sst_finetune/model_0.pt"
DATA_DIR="data/mt_dnn_uncased"
MODEL_DIR="checkpoints/${PREFIX}_eval_${TSTR}"
TWEET_DATA_DIR="../../../TweetScraper/Data/tweet/"
LOG_FILE="${MODEL_DIR}/log.log"
CUDA="True"

python3 eval.py --init_checkpoint ${BERT_PATH} --output_dir ${MODEL_DIR} --log_file ${LOG_FILE} --cuda ${CUDA} --tweet_data_dir ${TWEET_DATA_DIR}
