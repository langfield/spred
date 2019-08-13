# python3 train.py --train_dataset ../exchange/ETHUSDT_small.csv --gpst_model config.json --output_dir checkpoints --do_train --train_batch_size 1 --num_train_epochs 40
srun -J gpst -w vibranium --mem 10000 -c 4 python3 train.py --train_dataset ../exchange/ETHUSDT_small.csv --gpst_model config.json --output_dir checkpoints --do_train --train_batch_size 128 --num_train_epochs 30000
# python3 train.py --train_dataset ../exchange/ETHUSDT_small.csv --gpst_model config.json --output_dir checkpoints --do_train --train_batch_size 4 --num_train_epochs 1
