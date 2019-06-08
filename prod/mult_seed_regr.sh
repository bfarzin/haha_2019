#!/bin/bash

for n in {1..200};do \
    python regr.py \
	   --model-path '../data/rnn/' \
	   --sp-model '../all_tweets_es_0607_unigram' \
	   --load_enc 'twitter_es_enc_QRNN_0608_drop1_unigram_seed0' \
	   --split-seed 20190313 \
	   --qrnn 1\
    ;
done
