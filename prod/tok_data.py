from fastai.text import *
import sentencepiece as spm #https://github.com/google/sentencepiece
from sp_tok import *
import fire

def myreadlines(f, newline:str='\n', rsz:int=4096):
    "read with custom newline"          
    buf = ''
    while True: 
        while newline in buf:
            pos = buf.index(newline)
            yield buf[:pos]
            buf = buf[pos + len(newline):]
        chunk = f.read(rsz)
        if not chunk: break
        buf += chunk
        
def get_corpus_txt(corpus_fname = './data/all_file.txt'):
    corpus_txt = []
    with open(corpus_fname,'r') as f:
        next(f)
        for kk,line in enumerate(myreadlines(f,'"\n')): 
            corpus_txt.append(line.lstrip('"'))
    return corpus_txt

def build_sp_data(model_path:str, corpus_fname:str, out_pkl_name:str, sp_model:str,
                  vocab_size:int=60000, batch_size:int=64, verbose:bool=False, tmp_file_name:str="tmp_data"):
    PATH = Path(model_path)
    # corpus_fname = '../data/all_file.txt'
    # model_prefix = 'all_tweets_es_0509'
    corpus_txt = get_corpus_txt(corpus_fname)
    all_texts_df = pd.DataFrame(corpus_txt, columns=["tweet_text"])
    raw_text = all_texts_df.loc[:,'tweet_text']

    print("Default Rules:\n",[x.__name__ for x in default_rules],"\n\n")
    for rule in default_rules:
        raw_text = raw_text.apply(lambda x: rule(str(x)))    
    all_texts_df['new_text'] = 'xxbos ' + raw_text

    if verbose:
        print('Example of cleaned up text (original, then cleaned up)')
        for kk in range(3):
            print(all_texts_df['tweet_text'].iloc[kk])
            print(all_texts_df['new_text'].iloc[kk])
            print('-'*15)

    #tmp file for SP tokenizer
    formatted_text_file = tmp_file_name
    all_texts_df['new_text'].to_frame().to_csv(formatted_text_file, header=False,index=False,quotechar=' ')


    #user defined symbols for SP
    uds = [x for x in defaults.text_spec_tok if x != UNK]
    print(uds)

    spm.SentencePieceTrainer.Train(f'--input={formatted_text_file}'
                                   f' --model_prefix={sp_model}'
                                   f' --vocab_size={vocab_size}'
                                   f' --model_type=bpe'
                                   f" --user_defined_symbols={','.join(uds)}"
                                   f' --unk_piece={UNK} --bos_id=-1 --eos_id=-1 --pad_id=-1')

    mycust_tok = CustomTokenizer(SPTokenizer,sp_model,pre_rules=default_rules)
    sp_vocab = Vocab( get_itos(sp_model) )    

    #train/valid split
    all_texts = all_texts_df['new_text'].values.squeeze()
    idx = np.random.permutation(len(all_texts))
    cut = int(0.1 * len(idx))
    train_df = pd.DataFrame({'text':all_texts[idx[cut:]], 'labels':[0] * (len(all_texts)-cut)}, columns=['labels','text'])
    valid_df = pd.DataFrame({'text':all_texts[idx[:cut]], 'labels':[0] * cut}, columns=['labels','text'])

    print(f'Train:{train_df.shape}, valid:{valid_df.shape}')

    keyword_args = {'bs':batch_size}
    data = TextLMDataBunch.from_df(PATH, train_df, valid_df, 
                                   tokenizer=mycust_tok, vocab=sp_vocab,
                                   text_cols='text', label_cols='labels',**keyword_args)
    data.save(out_pkl_name)
    
if __name__ == "__main__":
    fire.Fire(build_sp_data)