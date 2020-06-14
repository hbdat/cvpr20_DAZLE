#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:25:09 2019

@author: war-machince
"""

import os,sys
pwd = os.getcwd()
sys.path.insert(0,pwd)
#%%
print('-'*30)
print(os.getcwd())
print('-'*30)
#%%
import pdb
import pandas as pd
import numpy as np
import gensim.downloader as api
import scipy.io as sio
import pickle
from global_setting_Pegasus import NFS_path
#%%
print('Loading pretrain w2v model')
model_name = 'word2vec-google-news-300'#best model
model = api.load(model_name)
dim_w2v = 300
print('Done loading model')
#%%
replace_word = [('-',' '),('eiffel','Eiffel')]
dataset = 'DeepFashion'
#%%
path = os.path.join(NFS_path,'data/{}/annotation.pkl'.format(dataset))
package=pickle.load(open(path,'rb'))
des = package['att_names']
#%% filter
#new_des = [' '.join(i.split('_')) for i in des]
#new_des = [' '.join(i.split('-')) for i in new_des]
#new_des = [' '.join(i.split('::')) for i in new_des]
#new_des = [i.split('(')[0] for i in new_des]
#new_des = [i[4:] for i in new_des]
#%% replace out of dictionary words
for pair in replace_word:
    for idx,s in enumerate(des):
        des[idx]=s.replace(pair[0],pair[1])
print('Done replace OOD words')
#%%
#df['new_des']=new_des
#df.to_csv('./attribute/CUB/new_des.csv')
#print('Done preprocessing attribute des')
#%%
counter_err = 0
all_w2v = []
for s in des:
#    print(s)
    words = s.split(' ')
    if words[-1] == '':     #remove empty element
        words = words[:-1]
    w2v = np.zeros(dim_w2v)
    for w in words:
        try:
            w2v += model[w]
        except Exception as e:
            print(e)
            counter_err += 1
    all_w2v.append(w2v[np.newaxis,:])
print('counter_err ',counter_err)
#%%
all_w2v=np.concatenate(all_w2v,axis=0)
pdb.set_trace()
#%%
with open('./w2v/{}_attribute.pkl'.format(dataset),'wb') as f:
    pickle.dump(all_w2v,f)    
