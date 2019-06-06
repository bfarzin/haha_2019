from set_seed import random_ctl
seed = random_ctl()

from fastai.text import *
import sentencepiece as spm #https://github.com/google/sentencepiece
from fastai.layers import LabelSmoothingCrossEntropy
import fire
from sp_tok import *
    
def build_lm(model_path:str, sp_model:str, data_pkl_name:str, enc_name:str, flat_loss:bool=True,
             qrnn:bool=True, n_hid:int=2304, n_epochs:int=25, dropmult:float=0.5, backward:bool=False):
    PATH = Path(model_path)
    defaults.text_spec_tok.append(NL) #add a New Line special char
    sp_vocab = Vocab( get_itos(sp_model) )    
    mycust_tok = CustomTokenizer(SPTokenizer, sp_model, pre_rules=default_rules)

    data = load_data(PATH,data_pkl_name)

    config = awd_lstm_lm_config.copy()
    config['qrnn'] = qrnn
    config['n_hid'] = n_hid
    print(config)
    learn = language_model_learner(data, AWD_LSTM, drop_mult=dropmult, pretrained=False, config=config)
    if flat_loss: learn.loss_func = FlattenedLoss(LabelSmoothingCrossEntropy)
    print(learn.model)
    learn.unfreeze()
    learn.fit_one_cycle(n_epochs, 4e-3, moms=(0.6,0.4), wd=0.02, pct_start=0.2)
    learn.save_encoder(enc_name)
    learn.save(f"twitter_es_{seed}{'_bwd' if backward else ''}")
    df_metrics = pd.DataFrame(np.array(learn.recorder.metrics),columns=learn.recorder.metrics_names)
    print(f"LM RndSeed: {seed},{df_metrics['accuracy'].max()}")
    
if __name__ == "__main__":
    fire.Fire(build_lm)