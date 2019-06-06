#!/bin/bash

## fit the model
python main.py \
       --model-path '../data/rnn/' \
       --sp-model '../all_tweets_es_0606' \
       --data-pkl-name 'tweet_es_lmdata_bwd.pkl' \
       --enc-name 'twitter_es_enc_QRNN_0606_seed0' \
       --flat_loss 1 \
       --qrnn 1 \
       --backward 1

for n in {1..100};do \
    python clas.py \
	   --model-path '../data/rnn/' \
	   --sp-model '../all_tweets_es_0606' \
	   --load_enc 'twitter_es_enc_QRNN_0606_seed0' \
	   --split-seed 20190313 \
	   --flat_loss 1 \
	   --qrnn 1\
	   --backward 1\
    ;
done
