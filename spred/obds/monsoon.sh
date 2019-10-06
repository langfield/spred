#!/bin/bash

HOURS="10"
TRUNC="50"
SAVE_PATH="../bookdfs/sampleset.csv"

python3 trainset.py --hours ${HOURS} --trunc ${TRUNC} --save_path ${SAVE_PATH}
