# python3 train.py --train_corpus ../exchange/ETHUSDT_small.csv --xlspred_model config.json --output_dir checkpoints --do_train --train_batch_size 4
python3 train.py --train_corpus ../exchange/ETHUSDT_small.csv --xlspred_model config.json --output_dir checkpoints --do_train --train_batch_size 4 --no_cuda --max_seq_length 16 --num_predict 4 --reuse_len 8 --data_batch_size 1
