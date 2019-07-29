# python3 train.py --train_corpus ../exchange/ETHUSDT_small.csv --xlspred_model config.json --output_dir checkpoints --do_train --train_batch_size 4
python3 train.py --train_corpus ../exchange/ETHUSDT_small.csv --xlspred_model config.json --output_dir checkpoints --do_train --train_batch_size 7 --no_cuda --max_seq_length 50 --num_predict 12 --reuse_len 25 --data_batch_size 1
