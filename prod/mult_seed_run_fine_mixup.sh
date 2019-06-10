#!/bin/bash

for n in {1..1};do \
    python clas_kfold_mixup.py \
	   --model-path '../data/rnn/' \
	   --sp-model '../all_tweets_es_0609' \
	   --load_enc 'twitter_es_enc_QRNN_0609_drop1_seed0_finetune' \
	   --split-seed 20190313 \
	   --flat-loss 1 \
	   --qrnn 1\
	   --wd 0.02\
	   --mixup 1\
	   --backward 0\
    ;
done
