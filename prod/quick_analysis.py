import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

'''
grep Clas out_bwd_tmp1?.txt > seed_class_data_bwd
'''
df = pd.read_csv('seed_class_data',names=['id','acc','f1'])
print(df.describe())
df['group'] = df.id.apply(lambda x: 0 if '10' in x else 1)
print(df.groupby('group').describe().transpose())


bins = np.linspace(0.84,0.86,30)
df[df['group']==0]['acc'].hist(alpha=0.5,label='seed0',bins=bins)
df[df['group']==1]['acc'].hist(alpha=0.5,label='seed1',bins=bins)
plt.axvline(df[df['group']==0]['acc'].mean(),c='blue',alpha=0.5)
plt.axvline(df[df['group']==1]['acc'].mean(),c='orange')
plt.title('Accuracy')
plt.legend()
plt.show()


'''
grep -A 30 '^epoch' out_tmp10.txt | grep '^[0-9]' > seed0_table.csv
'''
df = pd.read_csv('seed0_table.csv', delim_whitespace=True,
                 names=['e','tloss','vloss','acc','f1','time'])
df['new_group'] = df.e.diff()<0
df['new_group'] = df['new_group'].cumsum()
seed0_f1 = df.groupby('new_group')['f1'].max()[1:].astype(float)

df = pd.read_csv('seed1_table.csv', delim_whitespace=True,
                 names=['e','tloss','vloss','acc','f1','time'])
df['new_group'] = df.e.diff()<0
df['new_group'] = df['new_group'].cumsum()
seed1_f1 = df.groupby('new_group')['f1'].max()[1:].astype(float)


bins = np.linspace(0.80,0.82,30)
seed0_f1.hist(alpha=0.5,label='seed0',bins=bins)
seed1_f1.hist(alpha=0.5,label='seed1',bins=bins)
plt.axvline(seed0_f1.mean(),c='blue',alpha=0.5)
plt.axvline(seed1_f1.mean(),c='orange')

plt.legend()
plt.title("F1")
plt.show()

set_trace()
