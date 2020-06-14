# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:28:17 2019

@author: Warmachine
"""

import pandas as pd
from io import StringIO
import numpy as np
from global_setting_Local import NFS_path
import pickle
#%%
import pdb
#%%
anno_dir = NFS_path+'data/DeepFashion/'+'Anno/'
eval_dir = NFS_path+'data/DeepFashion/'+'Eval/'
#%%
unseen_classes = [12,15,2,19,29,30,26,44,42,48]
#%%
def convert_string_np(s,idx_e):
    if s is None:
        print(idx_e)
    list_int = [int(e) for e in s.split(' ') if e]
    return np.array(list_int).reshape((1,-1))

def read_text(file_path,sep): #Need to hardcode the logic that row is always tuple
    with open(file_path,'r') as f:
        rows = []
        for idx_l,line in enumerate(f):
            line = line.strip()
            if idx_l == 0:
                n = int(line)
            elif idx_l == 1:
                columns = [e for e in line.split(sep) if e]
            else:
                es = [e for e in line.split(sep) if e]
                row = []
                row.append(es[0])               #element 1
                row.append(' '.join(es[1:]))    #element 2
#                row[-1] = int(row[-1])
                rows.append(row)
        assert len(rows)==n
        df = pd.DataFrame(rows,columns =columns)
    return df
#%%
df_cat_name = read_text(anno_dir+'list_category_cloth.txt',sep = '  ')
df_cat_name['category_type'] = df_cat_name['category_type'].astype(int)

df_att_name = read_text(anno_dir+'list_attr_cloth.txt',sep = '  ')
df_att_name['attribute_type'] = df_att_name['attribute_type'].astype(int)
#%%
df_labels = read_text(anno_dir+'list_category_img.txt',sep = ' ')
df_labels['category_label'] = df_labels['category_label'].astype(int)
df_attr_anno = read_text(anno_dir+'list_attr_img.txt',sep = ' ')
df_split = read_text(eval_dir+'list_eval_partition.txt',sep = ' ')

df_join = df_attr_anno.join(df_labels.set_index('image_name'),on='image_name')
df_join = df_join.join(df_split.set_index('image_name'),on='image_name')
#%%
image_names = df_join['image_name'].values
cat_names = df_cat_name['category_name'].values
att_names = df_att_name['attribute_name'].values


labels = df_join['category_label'].values
labels = [convert_string_np(attr,idx_e) for idx_e,attr in enumerate(labels)]
labels = np.concatenate(labels,0)

attr_annos = df_join['attribute_labels'].values
attr_annos = [convert_string_np(attr,idx_e) for idx_e,attr in enumerate(attr_annos)]
attr_annos = np.concatenate(attr_annos,0)

split = df_join['evaluation_status'].values
#%%
classes = np.unique(labels)
att = np.zeros((len(df_cat_name),attr_annos.shape[1]))
for c in classes:
    mask = np.squeeze(labels == c)
    att[c,:] = np.mean(attr_annos[mask],0)
#%%
freq = df_labels['category_label'].value_counts()
df_cat_name = df_cat_name.join(freq)
#%%
df_cat_name.join(freq.to_frame())
#%%
df_unseen_cat = df_cat_name.iloc[unseen_classes]
#%%
test_unseen_loc =   np.array([i for i in range(labels.shape[0]) if labels[i] in unseen_classes])
test_seen_loc =     np.array([i for i in range(labels.shape[0]) if (labels[i] not in unseen_classes) and (split[i]=='test')])
train_loc =         np.array([i for i in range(labels.shape[0]) if (labels[i] not in unseen_classes) and (split[i]!='test')])
assert len(train_loc)+len(test_seen_loc)+len(test_unseen_loc) == len(split)
#%%
package = {'image_names':image_names,'cat_names':cat_names,'att_names':att_names,'train_loc':train_loc,'test_seen_loc':test_seen_loc,'test_unseen_loc':test_unseen_loc,'att':att,'labels':labels}
pickle.dump(package, open( NFS_path+"data/DeepFashion/annotation.pkl", "wb" ) )