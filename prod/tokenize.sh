#!/bin/bash

## build the tokenizer dataset
python tok_data.py \
       --model-path '../data/rnn/' \
       --corpus-fname '../data/all_file.txt'\
       --out-pkl-name 'tweet_es_lmdata_bwd.pkl'\
       --sp-model '../all_tweets_es_0606' \
       --vocab-size 60000 \
       --batch-size 64 \
       --verbose 1 \
       --backward 1
