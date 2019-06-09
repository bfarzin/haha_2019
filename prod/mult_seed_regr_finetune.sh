#!/bin/bash

for n in {1..20};do \
    python regr_kfold.py \
	   --model-path '../data/rnn/' \
	   --sp-model '../all_tweets_es_0606' \
	   --load_enc 'twitter_es_enc_QRNN_0608_drop1_seed0_finetune' \
	   --split-seed 20190313 \
	   --qrnn 1\
    ;
done
