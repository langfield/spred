rm -rf checkpoints/

# python3 train.py --train_corpus ../exchange/ETHUSDT_small.csv --xlspred_model config.json --output_dir checkpoints --do_train --train_batch_size 4

python3 train.py --train_corpus ../exchange/ETHUSDT_small.csv --xlspred_model config.json --output_dir checkpoints --do_train --train_batch_size 4 --max_seq_length 50 --num_predict 12 --reuse_len 25 --data_batch_size 1 --num_train_epochs 1
#srun -J xlnetpt --mem 10000 -c 12 -w vanadium python3 train.py --train_corpus ../exchange/ETHUSDT_small.csv --xlspred_model config.json --output_dir checkpoints --do_train --train_batch_size 4 --max_seq_length 50 --num_predict 12 --reuse_len 25 --data_batch_size 1 --num_train_epochs 50
