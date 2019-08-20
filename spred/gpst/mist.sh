#!/bin/bash

if [ "$(whoami)" != "mckade" ]; then
    python3 eval.py --batch 1 --width 50 --input "../exchange/concatenated_price_data/ETHUSDT_drop.csv" --output_dir "graphs/" --terminal_plot_width 50 --stationarize
else
    python3 eval.py --batch 1 --width 50 --input "../../../ETHUSDT_drop.csv" --output_dir "graphs/" --terminal_plot_width 50 --stationarize
fi

