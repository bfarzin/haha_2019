#!/bin/bash

for n in {1..1};do \
    python regr_kfold_mixup.py \
	   --model-path '../data/rnn/' \
	   --sp-model '../all_tweets_es_0609' \
	   --load_enc 'twitter_es_enc_QRNN_0609_drop1_seed0_finetune' \
	   --wd 0.1\
	   --mixup 1\
	   --split-seed 20190313 \
	   --qrnn 1\
    ;
done
