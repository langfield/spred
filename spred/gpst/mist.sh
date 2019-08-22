#!/bin/bash

if [ "$(whoami)" != "mckade" ]; then
    python3 eval.py --eval_batch_size 1 --width 50 --dataset "../exchange/concatenated_price_data/ETHUSDT_drop.csv" --graph_dir "graphs/" --terminal_plot_width 50 --normalize
else
    python3 eval.py --eval_batch_size 1 --width 50 --dataset "../../../ETHUSDT_drop.csv" --graph_dir "graphs/" --terminal_plot_width 50 --stationarize
fi

