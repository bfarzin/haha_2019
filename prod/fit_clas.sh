#!/bin/bash

python clas.py \
       --model-path '../data/rnn/' \
       --sp-model '../__all_tweets_es_0521' \
       --load-enc '__twitter_es_enc_QRNN_0521_labelsmooth'\
       --split-seed 20190313 \ #for clas train/valid split. Hold constatnt
       --gpu-id 0 \
       --flat_loss 1 \
       --qrnn 1
       
