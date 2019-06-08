#!/bin/bash

for n in {1..61};do \
    python regr.py \
	   --model-path '../data/rnn/' \
	   --sp-model '../all_tweets_es_0607' \
	   --load_enc 'twitter_es_enc_QRNN_0607_drop1_seed0' \
	   --split-seed 20190313 \
	   --qrnn 1\
    ;
done
