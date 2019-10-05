#!/bin/bash

PRINT_STATS="True"
HOUR="0"
DEPTH="-1"
BOUND="100.0"
ANALYSIS_DEPTH="10"
NUM_GAP_FREQ_LEVELS="3"
NUM_GAP_FREQ_SIZES="5"
FIG_DIR="figs/"

COMPUTE_K="False"
HOURS="100"
SIGMA="3"

python3 format.py --print_stats ${PRINT_STATS} --hour ${HOUR} --depth ${DEPTH} --bound ${BOUND} --analysis_depth ${ANALYSIS_DEPTH} --num_gap_freq_levels ${NUM_GAP_FREQ_LEVELS} --num_gap_freq_sizes ${NUM_GAP_FREQ_SIZES} --fig_dir ${FIG_DIR} --compute_k ${COMPUTE_K} --hours ${HOURS} --sigma ${SIGMA}
