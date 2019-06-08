#!/bin/bash

## build the tokenizer dataset
python tok_data.py \
       --model-path '../data/rnn/' \
       --corpus-fname '../data/all_file.txt'\
       --out-pkl-name 'tweet_es_lmdata_bs256_unigram.pkl'\
       --sp-model '../all_tweets_es_0607_unigram' \
       --vocab-size 60000 \
       --batch-size 256 \
       --verbose 1 \
       --backward 0 \
       --model-type 'unigram'

## fit the model
python main.py \
       --model-path '../data/rnn/' \
       --sp-model '../all_tweets_es_0607' \
       --data-pkl-name 'tweet_es_lmdata_bs256_unigram.pkl' \
       --batch-size 64 \
       --enc-name 'twitter_es_enc_QRNN_0608_drop1_unigram_seed0' \
       --flat_loss 1 \
       --qrnn 1 \
       --n-epochs 10 \
       --dropmult 1.0 \
       --wd 0.1 \
       --backward 0
