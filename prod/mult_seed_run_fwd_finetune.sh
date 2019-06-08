#!/bin/bash

for n in {1..2};do \
    python clas_kfold.py \
	   --model-path '../data/rnn/' \
	   --sp-model '../all_tweets_es_0606' \
	   --load_enc 'twitter_es_enc_QRNN_0608_drop1_seed0_finetune' \
	   --split-seed 20190313 \
	   --flat-loss 1 \
	   --qrnn 1\
	   --backward 0\
    ;
done
