# python3 train.py --train_corpus ../exchange/ETHUSDT_small.csv --xlspred_model config.json --output_dir checkpoints --do_train --train_batch_size 4
# python3 simple_lm_finetuning.py --train_corpus ../sample_text/sample_text.txt --bert_model bert-base-uncased --output_dir checkpoints --do_train --train_batch_size 3 --no_cuda --max_seq_length 20
python3 simple_lm_finetuning.py --train_corpus ../sample_text/sample_text.txt --bert_model bert-base-uncased --output_dir checkpoints --do_train --train_batch_size 3 --max_seq_length 20
