#!/bin/bash

## build the tokenizer dataset
python tok_data.py \
       --model-path '../data/rnn/' \
       --corpus-fname '../data/all_file.txt'\
       --out-pkl-name 'tweet_es_lmdata_test.pkl'\
       --sp-model '../__all_tweets_es_0521' \
       --vocab-size 60000 \
       --batch-size 64 \
       --verbose 1
## fit the model
python main.py \
       --model-path '../data/rnn/' \
       --sp-model '../__all_tweets_es_0521' \
       --data-pkl-name 'tweet_es_lmdata_test.pkl' \
       --gpu-id 0 \
       --flat_loss 1 \
       --qrnn 1
