import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

'''
grep Clas out_tmp1?.txt > seed_class_data
'''
df = pd.read_csv('seed_regr_data',names=['id','acc'])
print(df.describe())
df['group'] = df.id.apply(lambda x: 0 if '10' in x else 1)
print(df.groupby('group').describe().transpose())


bins = np.linspace(0.53,0.59,30)
df[df['group']==0]['acc'].hist(alpha=0.5,label='seed0',bins=bins)
df[df['group']==1]['acc'].hist(alpha=0.5,label='seed1',bins=bins)
plt.axvline(df[df['group']==0]['acc'].mean(),c='blue',alpha=0.5)
plt.axvline(df[df['group']==1]['acc'].mean(),c='orange')
plt.title('MSE')
plt.legend()
plt.show()