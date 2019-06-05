# haha_2019 production scripts

`CUDA_VISIBLE_DEVICES=0 ./mult_seed_run.sh | tee --append out_tmp10.txt`


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
  * Average outputs for classifer softmax before making choice (single seeds might be better, but not many possible entries)
  * [SMOTE to balance classes](https://jair.org/index.php/jair/article/view/10302)


3. Ideas:
  * BiDir LSTM for classification?  And for LM?
  * Backward and averaging?
  * Back Translate ES-EN-ES to get new version (And at test time)