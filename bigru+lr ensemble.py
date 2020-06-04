import pandas as pd

bigru = '../submissions/bigrusubm.csv'
lr = '../submissions/lr_hash.csv'

p_bigru = pd.read_csv(bigru)
p_lr = pd.read_csv(lr)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
p_res = p_bigru.copy()
p_res[label_cols] = (p_lr[label_cols] + p_bigru[label_cols]) / 2

p_res.to_csv('subm_bigu+lr.csv', index=False)