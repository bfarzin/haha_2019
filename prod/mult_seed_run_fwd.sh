#!/bin/bash

for n in {1..100};do \
    python clas.py \
	   --model-path '../data/rnn/' \
	   --sp-model '../all_tweets_es_0607_unigram' \
	   --load_enc 'twitter_es_enc_QRNN_0608_drop1_unigram_seed0' \
	   --split-seed 20190313 \
	   --flat-loss 1 \
	   --qrnn 1\
	   --backward 0\
    ;
done
