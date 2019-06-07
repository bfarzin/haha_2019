# haha_2019 production scripts

Check if you want to run forward or backward on tokenization (run both and average results)
1. ~~`./tokenize.sh | tee out_tokenizer.txt`~~  `CUDA_VISIBLE_DEVICES=0 ./fit_lm.sh`
2. `CUDA_VISIBLE_DEVICES=0 ./mult_seed_run_bwd.sh | tee --append out_bwd_tmp10.txt`
3. Change the seed from `seed0` -> `seed1`, and re-run above scripts to generate LM and fits
  1. don't need to gen data again, go right to `main.py` call with:
  ```
  CUDA_VISIBLE_DEVICES=1 python main.py --model-path '../data/rnn/' --sp-model '../all_tweets_es_0606' --data-pkl-name 'tweet_es_lmdata_bwd.pkl' --enc-name 'twitter_es_enc_QRNN_0606_bwd_seed1' --flat_loss 1 --qrnn 1 --backward 1 --n-epochs 5 --dropout 0.75
  ```
Original:
*. `CUDA_VISIBLE_DEVICES=0 ./mult_seed_run.sh | tee --append out_tmp10.txt`


## Paper Outline

1. ULMFiT method:
  * Build LM with Twitter data
  * Tear off head, replace with AdaptivePooling and Classifer
  * Split train/validate/test data set
  * Fit with random seeds

2. Innovations
  * encoded `\n` as `xxnl` to include new lines
  * vocab based on twitter language used (475k tweets in spanish) Better coverage than Wiki103 or standard vocab (including emoticons)
  * use of sentence piece to reduce the out of vocab (OOV) lanugage
  * Two random seeds for LM fit
  * Random seeds on each of the classifier fits (100 each LM rand seed)
    * This was originally a test, done with a single epoch on LM (so not the best) and then 100 runs to test if there was variation
  * Average outputs for classifer softmax before making choice (single seeds might be better, but not many possible entries)
  * [SMOTE to balance classes](https://jair.org/index.php/jair/article/view/10302)


3. Ideas:
  ~~* BiDir LSTM for classification?  And for LM?~~
  * Backward and averaging?
  * Back Translate ES-EN-ES to get new version (And at test time)
  * Higher Dropmult for LM (closer to 1.) and train for 4 epochs?
  * Select on F1 rather than Accuracy?  (becuase that is comp metric)
  * Does the top-half of models in the acc distribution land in the top-half of the test distribution?  Or it is all over the place?