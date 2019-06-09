from set_seed import random_ctl
seed = random_ctl()

from fastai.text import *
from fastai.callbacks import SaveModelCallback
from fastai.layers import LabelSmoothingCrossEntropy

import sentencepiece as spm #https://github.com/google/sentencepiece
import fire

from sp_tok import *
from sklearn.model_selection import KFold

def split_data_by_idx(all_texts_df:DataFrame, train_idx, valid_idx):
    df_train = all_texts_df.iloc[train_idx,:]
    df_valid     = all_texts_df.iloc[valid_idx,:]

    return df_train, df_valid
    
    
def fit_clas(model_path:str, sp_model:str,
             qrnn:bool=True, n_hid:int=2304, load_enc:str=None, split_seed:int=None):
    PATH = Path(model_path)
    # torch.backends.cudnn.enabled=False
    
    defaults.text_spec_tok.append(NL) #add a New Line special char
    sp_vocab = Vocab( get_itos(sp_model) )    
    mycust_tok = CustomTokenizer(SPTokenizer,sp_model,pre_rules=default_rules)

    all_texts_df = pd.read_csv('../data/haha_2019_train.csv')
    all_texts_df.funniness_average.fillna(0,inplace=True)
    raw_text = all_texts_df.loc[:,'text']

    print("Default Rules:\n",[x.__name__ for x in default_rules],"\n\n")
    for rule in default_rules: raw_text = raw_text.apply(lambda x: rule(str(x)))
    all_texts_df['new_text'] = raw_text #databunch adds `xxbos` so don't add here

    kfolder = KFold(n_splits=5, random_state=split_seed, shuffle=True)
    for n_fold, (train_idx,valid_idx) in enumerate(kfolder.split(all_texts_df)):
        df_train,df_valid = split_data_by_idx(all_texts_df,train_idx,valid_idx)
    
        data = TextClasDataBunch.from_df(PATH,df_train,df_valid,
                                   tokenizer=mycust_tok, vocab=sp_vocab,
                                   text_cols='new_text', label_cols='funniness_average')
        config = awd_lstm_clas_config.copy()
        config['qrnn'] = qrnn
        config['n_hid'] = n_hid
        print(config)
        learn = text_classifier_learner(data, AWD_LSTM, drop_mult=0.7,pretrained=False,config=config)
        if load_enc : learn.load_encoder(load_enc)
        learn.unfreeze()

        learn.fit_one_cycle(10, slice(1e-2/(2.6**4),1e-2), moms=(0.7,0.4), pct_start=0.25, div_factor=8.,
                            callbacks=[SaveModelCallback(learn,every='improvement',mode='min',
                                                         name='best_vloss_model_Q')])
        learn.save(f'haha_regr_0609_fld{n_fold}_{seed}')
        print(f"Reg Fold: {n_fold} RndSeed: {seed},{min(learn.recorder.val_losses)}")
    
if __name__ == "__main__":
    fire.Fire(fit_clas)
