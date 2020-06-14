#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 17:14:16 2019

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
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models.resnet as models
from PIL import Image
import h5py
import numpy as np
import scipy.io as sio
import pickle
from global_setting_Pegasus import NFS_path

#%%
import pdb
#%%
idx_GPU = 4
is_save = True
dataset = 'DeepFashion'
#%%
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
#%%
img_dir = os.path.join(NFS_path,'data/{}/'.format(dataset))
file_paths = os.path.join(NFS_path,'data/{}/annotation.pkl'.format(dataset))
save_path = os.path.join(NFS_path,'data/{}/feature_map_ResNet_101_{}_sep_seen_samples.hdf5'.format(dataset,dataset))
attribute_path = './w2v/{}_attribute.pkl'.format(dataset)
#pdb.set_trace()
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Batch size for training (change depending on how much memory you have)
batch_size = 32

device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")
#%%

model_ref = models.resnet101(pretrained=True)
model_ref.eval()

model_f = nn.Sequential(*list(model_ref.children())[:-2])
model_f.to(device)
model_f.eval()

for param in model_f.parameters():
    param.requires_grad = False
#%%
class CustomedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir , file_paths, transform=None):
        self.package = pickle.load(open(file_paths,'rb'))
        self.image_files = self.package['image_names']
        self.att = self.package['att']
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_file = os.path.join(self.img_dir,image_file)
        image = Image.open(image_file)
        if image.mode == 'L':
            image=image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

#%%
input_size = 224
data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
dataset = CustomedDataset(img_dir , file_paths, data_transforms)
dataset_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)
#%%
with torch.no_grad():
    all_features = []
    for i_batch, imgs in enumerate(dataset_loader):
        print(i_batch)
        imgs=imgs.to(device)
        features = model_f(imgs)
        all_features.append(features.cpu().numpy())
    all_features = np.concatenate(all_features,axis=0)
#%% get remaining metadata
labels = dataset.package['labels'].squeeze()

train_loc = dataset.package['train_loc'].squeeze()#--> train_feature = TRAIN SEEN
test_seen_loc = dataset.package['test_seen_loc'].squeeze()
test_unseen_loc = dataset.package['test_unseen_loc'].squeeze()
att = dataset.att
#%%
features_train = all_features[train_loc]
label_train = labels[train_loc]
img_train = dataset.image_files[train_loc].astype('S')

features_test_seen = all_features[test_seen_loc]
label_test_seen = labels[test_seen_loc]
img_test_seen = dataset.image_files[test_seen_loc].astype('S')

features_test_unseen = all_features[test_unseen_loc]
label_test_unseen = labels[test_unseen_loc]
img_test_unseen = dataset.image_files[test_unseen_loc].astype('S')
#%% construct attribute w2v
with open(attribute_path,'rb') as f:
    w2v_att = pickle.load(f)
assert w2v_att.shape == (1000,300)
print('load w2v_att')
#%%
string_dt = h5py.special_dtype(vlen=str)
if is_save:
    f = h5py.File(save_path, "w")
    
#    f.create_dataset('feature_map_train', data=features_train,compression="gzip")
#    f.create_dataset('label_train', data=label_train,compression="gzip")
#    f.create_dataset('img_train', data=img_train,dtype = string_dt,compression="gzip")
    pdb.set_trace()
    unique_label_train = np.squeeze(np.unique(label_train))
    for l in unique_label_train:
        mask_l = label_train == l
        print(l,np.sum(mask_l))
        f.create_dataset('feature_map_train/{}'.format(l), data=features_train[mask_l],compression="gzip")
        f.create_dataset('label_train/{}'.format(l), data=label_train[mask_l],compression="gzip")
        f.create_dataset('img_train/{}'.format(l), data=img_train[mask_l],dtype = string_dt,compression="gzip")
    
    f.create_dataset('feature_map_test_seen', data=features_test_seen,compression="gzip")
    f.create_dataset('label_test_seen', data=label_test_seen,compression="gzip")
    f.create_dataset('img_test_seen', data=img_test_seen,dtype = string_dt,compression="gzip")
    
    f.create_dataset('feature_map_test_unseen', data=features_test_unseen,compression="gzip")
    f.create_dataset('label_test_unseen', data=label_test_unseen,compression="gzip")
    f.create_dataset('img_test_unseen', data=img_test_unseen,dtype = string_dt,compression="gzip")
    
    f.create_dataset('labels', data=labels,compression="gzip")
    f.create_dataset('imgs', data=dataset.image_files,dtype = string_dt,compression="gzip")
    f.create_dataset('train_loc', data=train_loc,compression="gzip")
    f.create_dataset('test_seen_loc', data=test_seen_loc,compression="gzip")
    f.create_dataset('test_unseen_loc', data=test_unseen_loc,compression="gzip")
    f.create_dataset('att', data=att,compression="gzip")
    f.create_dataset('w2v_att', data=w2v_att,compression="gzip")
    f.close()
pdb.set_trace()