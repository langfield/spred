#!/bin/bash

SOURCE_DIR="/root/books/"
HOURS="5"
TRUNC="50"
SAVE_PATH="/root/books/sampleset.csv"

python3 trainset.py --hours ${HOURS} --trunc ${TRUNC} --save_path ${SAVE_PATH} --source_dir ${SOURCE_DIR}
