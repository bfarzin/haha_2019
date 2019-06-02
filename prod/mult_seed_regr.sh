#!/bin/bash

# ## fit the model
# python main.py \
#        --model-path '../data/rnn/' \
#        --sp-model '../__all_tweets_es_0521' \
#        --data-pkl-name 'tweet_es_lmdata_test.pkl' \
#        --enc-name 'twitter_es_enc_QRNN_0520_seed0' \
#        --flat_loss 1 \
#        --qrnn 1 \
#        --n-epochs 1

for n in {1..100};do \
    python regr.py \
	   --model-path '../data/rnn/' \
	   --sp-model '../__all_tweets_es_0521' \
	   --load_enc 'twitter_es_enc_QRNN_0520_seed1' \
	   --split-seed 20190313 \
	   --qrnn 1\
    ;
done
