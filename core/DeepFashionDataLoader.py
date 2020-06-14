# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 10:31:33 2019

@author: Warmachine
"""

import os,sys
#import scipy.io as sio
import torch
import torch.nn.functional as F  
import numpy as np
import h5py
import time
import math
from sklearn import preprocessing
from global_setting import NFS_path
#%%
import scipy.io as sio
import pandas as pd
import pickle 
#%%
#import pdb
#%%
img_dir = os.path.join(NFS_path,'data/DeepFashion/')
anno_path = os.path.join(NFS_path,'data/DeepFashion/annotation.pkl')

class DeepFashionDataLoader():
    def __init__(self, data_path, device, is_balance =True, is_scale = False,verbose = False):

        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.device = device
        self.dataset = 'DeepFashion'
        print('$'*30)
        print(self.dataset)
        print('$'*30)
        self.datadir = self.data_path + 'data/{}/'.format(self.dataset)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        self.verbose = verbose
        self.n_reuse = 2
        self.read_matdataset()
        
        self.seeker = np.zeros(self.ntrain_class)
        ### setup balance training ###
        if self.is_balance:
            print('Balance dataloader')
        else:
            print('No balance adjustment')
        self.cur_classes_idx = 0
        
        self.idx_part = 0
        self.part_size = 10000
        self.idx_b = 0
        self.part_features = None
        self.part_labels = None
        
        print('Partition size {}'.format(self.part_size))
        
        self.convert_new_classes()
        print('Excluding non-sample classes')
        
        print('-'*30)
        print('DeepFashion')
        print('-'*30)
        
    def reset_seeker(self):
        self.seeker[:] = 0
        
    def get_class(self):
        self.cur_classes_idx = (self.cur_classes_idx+1)%self.ntrain_class
        
        return self.train_class[self.cur_classes_idx].cpu()
    
    def augment_img_path(self,anno_path=anno_path,img_dir=img_dir):
        self.package = pickle.load(open(anno_path,'rb'))
        self.image_files = self.package['image_names']
        self.cat_names = self.package['cat_names']
        
        def convert_path(image_files,img_dir):
            new_image_files = []
            for idx in range(len(image_files)):
                image_file = image_files[idx]
                image_file = os.path.join(img_dir,image_file)
                new_image_files.append(image_file)
            return np.array(new_image_files)
        self.image_files = convert_path(self.image_files,img_dir)
    
        path= self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        hf = h5py.File(path, 'r')
    
        trainval_loc = np.array(hf.get('trainval_loc'))
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))
    
        self.data['train_seen']['img_path'] = None#self.image_files[trainval_loc]
        self.data['test_seen']['img_path'] = self.image_files[test_seen_loc]
        self.data['test_unseen']['img_path'] = self.image_files[test_unseen_loc]
    
        self.attr_name = self.package['att_names']
    
    def next_batch_img(self, batch_size,class_id,is_trainset = False):
        features = None
        labels = None
        img_files = None
        if class_id in self.seenclasses:
            if is_trainset:
                features = self.data['train_seen']['resnet_features']
                labels = self.data['train_seen']['labels']
                img_files = self.data['train_seen']['img_path']
            else:
                features = self.data['test_seen']['resnet_features']
                labels = self.data['test_seen']['labels']
                img_files = self.data['test_seen']['img_path']
        elif class_id in self.unseenclasses:
            features = self.data['test_unseen']['resnet_features']
            labels = self.data['test_unseen']['labels']
            img_files = self.data['test_unseen']['img_path']
        else:
            raise Exception("Cannot find this class {}".format(class_id))
    
        #note that img_files is numpy type !!!!!
    
        idx_c = torch.squeeze(torch.nonzero(labels == class_id))
    
        features = features[idx_c]
        labels = labels[idx_c]
        img_files = img_files[idx_c.cpu().numpy()]
    
        batch_label = labels[:batch_size].to(self.device)
        batch_feature = features[:batch_size].to(self.device)
        batch_files = img_files[:batch_size]
        batch_att = self.att[batch_label].to(self.device)
    
        return batch_label, batch_feature,batch_files, batch_att
    
    def sample_from_parition(self):
        print('load data from hdf')     # this is slow because it loads non-consecutive memory block. Thus it may need to traverse the whole file to load stuff.
        
        self.part_features = []
        self.part_labels = []
        target_class_size = self.part_size//self.ntrain_class
        for idx_l,l in enumerate(self.data['train_seen']['labels']):
            
            features_c = self.data['train_seen']['resnet_features_hdf'][l]
            n_samples_c = len(features_c)
            if self.seeker[idx_l]*target_class_size >= n_samples_c:        ## seek to different partition of data
                self.seeker[idx_l] = 0
            start_seek = self.seeker[idx_l]*target_class_size
            end_seek = min(n_samples_c,(self.seeker[idx_l]+1)*target_class_size)
            idx_samples_c = np.arange(start_seek,end_seek).tolist()#np.random.choice(n_samples_c,size = samples_class_size)
            
            n_select_c = len(idx_samples_c)
            
            print(l,end='..')
            
            
            self.part_features.append(features_c[idx_samples_c])        ## only work with list datatype
            part_labels = [int(l)]*n_select_c
            part_labels = self.map_old2new_classes[part_labels]
            self.part_labels.append(part_labels)
            self.seeker[idx_l] += 1
        print()
    
    def next_batch(self, batch_size):
        if self.is_balance:
            if self.idx_b == 0:
                tic = time.clock()
                self.sample_from_parition()
                print('Elapsed time {}'.format(time.clock()-tic))
            
            batch_feature = []
            batch_label = []
            
            n_target_class = max(batch_size //self.ntrain_class,1)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class),min(self.ntrain_class,batch_size),replace=False).tolist()
            for i_c in sampled_idx_c:
                part_feature = self.part_features[i_c]
                part_label = self.part_labels[i_c]
                n_samples_c = part_feature.shape[0]
                idx_sample_c = np.random.choice(n_samples_c,n_target_class)
                batch_feature.append(part_feature[idx_sample_c])
                batch_label.append(part_label[idx_sample_c])
            
            
            
#            idx_samples_b = torch.randperm(self.part_labels.size(0))[0:batch_size]
#            batch_feature = self.part_features[idx_samples_b].to(self.device)
#            batch_label =  self.part_labels[idx_samples_b].to(self.device)
            batch_feature = torch.from_numpy(np.concatenate(batch_feature)).to(self.device)
            batch_label= torch.from_numpy(np.concatenate(batch_label)).to(self.device)
            batch_att = self.att[batch_label].to(self.device)
            
            ##increment
            self.idx_b = (self.idx_b + 1)%(self.part_size//batch_size*self.n_reuse)
            
        else:
            raise Exception('Not Implemented')
            if self.idx_b == 0:
#                idx_samples = (np.arange(self.part_size)+self.idx_part*self.part_size).tolist()
                self.part_features = torch.tensor(self.data['train_seen']['resnet_features_hdf'][self.idx_part*self.part_size:(self.idx_part+1)*self.part_size])
                self.part_labels = self.data['train_seen']['labels'][self.idx_part*self.part_size:(self.idx_part+1)*self.part_size]
                ## permute
                idx_permute = torch.randperm(self.part_features.size(0))
                self.part_features = self.part_features[idx_permute]
                self.part_labels = self.part_labels[idx_permute]
                
                ## increment
                self.idx_part = (self.idx_part+1)%(self.ntrain//self.part_size)
        
            batch_feature = self.part_features[self.idx_b*batch_size:(self.idx_b+1)*batch_size].to(self.device)
            batch_label =  self.part_labels[self.idx_b*batch_size:(self.idx_b+1)*batch_size].to(self.device)
            batch_att = self.att[batch_label].to(self.device)
            
            ##increment
            self.idx_b = (self.idx_b + 1)%(self.part_labels.size(0)//batch_size)
            
        if self.verbose:
            print('unique labels in batch {}'.format(torch.unique(batch_label)))
            
        return batch_label, batch_feature, batch_att

    def convert_new_classes(self):
        self.att = self.att[self.available_classes]
        self.att = F.normalize((self.att+1)/2)
        
#        self.data['train_seen']['labels'] = self.map_old2new_classes[self.data['train_seen']['labels']]
        self.data['test_seen']['labels'] = self.map_old2new_classes[self.data['test_seen']['labels']]
        self.data['test_unseen']['labels'] = self.map_old2new_classes[self.data['test_unseen']['labels']]
        
        self.seenclasses = self.map_old2new_classes[self.seenclasses].to(self.device)#torch.unique(self.data['train_seen']['labels']).to(self.device)
        self.unseenclasses = torch.unique(self.data['test_unseen']['labels']).to(self.device)
#        self.ntrain = self.data['train_seen']['labels'].size()[0]
#        self.ntrain_class = self.seenclasses.size(0)
#        self.ntest_class = self.unseenclasses.size(0)
#        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        
    def read_matdataset(self):

        path= self.datadir + 'feature_map_ResNet_101_{}_sep_seen_samples.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        tic = time.clock()
        hf = h5py.File(path, 'r')
        
#        labels = np.array(hf.get('labels'))
        
        att = np.array(hf.get('att'))
        
        ## remap classes this is because there is some classes that does not have training sample
        self.available_classes = np.where(np.sum(att,axis = 1)!=0)[0]
        self.map_old2new_classes = np.ones(att.shape[0])*-1
        self.map_old2new_classes[self.available_classes] = np.arange(self.available_classes.shape[0])
        self.map_old2new_classes = torch.from_numpy(self.map_old2new_classes).long()
        ##
        
        self.att = torch.from_numpy(att).float().to(self.device)
        
        w2v_att = np.array(hf.get('w2v_att'))
        self.w2v_att = torch.from_numpy(w2v_att).float().to(self.device)
        
        labels = hf['label_train']
        seenclasses = [int(l) for l in labels]
        n_sample_classes = [len(labels[str(l)]) for l in seenclasses]
        
        
        test_unseen_label = torch.from_numpy(np.array(hf.get('label_test_unseen'),dtype=np.int32)).long()#.to(self.device)
        test_seen_label = torch.from_numpy(np.array(hf.get('label_test_seen'),dtype=np.int32)).long()#.to(self.device)
        
        self.seenclasses = torch.tensor(seenclasses)
        self.unseenclasses = torch.unique(test_unseen_label).to(self.device)
        self.ntrain = sum(n_sample_classes)
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        ## containing missing classes therefore cannot determine the set of all available label
#        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        
        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features_hdf'] = hf['feature_map_train']
        self.data['train_seen']['labels']= labels

#        input('Debug version b')
        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None
        
        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = torch.from_numpy(np.array(hf.get('feature_map_test_seen'),dtype=np.float32)).float()
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = torch.from_numpy(np.array(hf.get('feature_map_test_unseen'),dtype=np.float32)).float()
        self.data['test_unseen']['labels'] = test_unseen_label

        print('Finish loading data in ',time.clock()-tic)