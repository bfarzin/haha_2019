#!/bin/bash

## fit the model
python main.py \
       --model-path '../data/rnn/' \
       --sp-model '../__all_tweets_es_0521' \
       --data-pkl-name 'tweet_es_lmdata_test.pkl' \
       --enc-name 'twitter_es_enc_QRNN_0520_seed0' \
       --gpu-id 1 \
       --flat_loss 1 \
       --qrnn 1 \
       --n-epochs 1

python clas.py \
       --model-path '../data/rnn/' \
       --sp-model '../__all_tweets_es_0521' \
       --load_enc 'twitter_es_enc_QRNN_0520_seed0' \
       --split-seed 20190313 \
       --gpu-id 1 \
       --flat_loss 1 \
       --qrnn 1

