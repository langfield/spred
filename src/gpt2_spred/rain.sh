# python3 train.py --train_dataset ../exchange/ETHUSDT_small.csv --gptspred_model config.json --output_dir checkpoints --do_train --train_batch_size 1 --num_train_epochs 40
# srun -J gpt -w vibranium --mem 30000 -c 8 python3 train.py --train_dataset ../exchange/ETHUSDT_small.csv --gptspred_model config.json --output_dir checkpoints --do_train --train_batch_size 4 --num_train_epochs 1
python3 train.py --train_dataset ../exchange/ETHUSDT_small.csv --gptspred_model config.json --output_dir checkpoints --do_train --train_batch_size 4 --num_train_epochs 1
