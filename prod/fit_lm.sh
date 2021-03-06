#!/bin/bash

## build the tokenizer dataset
python tok_data.py \
       --model-path '../data/rnn/' \
       --corpus-fname '../data/all_file.txt'\
       --out-pkl-name 'tweet_es_lmdata.pkl'\
       --sp-model '../all_tweets_es_0606' \
       --vocab-size 60000 \
       --batch-size 64 \
       --verbose 1 \
       --backward 0 \
       --model-type 'bpe'

## fit the model
python main.py \
       --model-path '../data/rnn/' \
       --sp-model '../all_tweets_es_0606' \
       --data-pkl-name 'tweet_es_lmdata.pkl' \
       --enc-name 'twitter_es_enc_QRNN_0606_seed0' \
       --split-seed 20190313 \
       --flat_loss 1 \
       --qrnn 1 \
       --n-epochs 5 \
       --dropmult 1.0 \
       --backward 0
