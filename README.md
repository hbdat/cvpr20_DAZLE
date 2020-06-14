# Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention

## Overview
This repository contains the implementation of [Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention](http://khoury.neu.edu/home/eelhami/publications/FineGrainedZSL-CVPR20.pdf).
> In this work, we develop a zero-shot fine-grained recognition with the ability to localize attributes using a dense attribute-based attention and embedding mechanism.

---
## Prerequisites
+ Python 3.x
+ PyTorch 1.x.x
+ sklearn
+ matplotlib
+ skimage
+ scipy

---
## Data Preparation
1) Please download and extract information into the `./data folder`. We include details about download links as well as what are they used for in each folder within `./data folder`.

2) **[Optional]** For DeepFashion dataset, we partition seen/unseen classes and training/testing split via:
```
python ./extract_feature/extract_annotation_DeepFashion.py							#create ./data/DeepFashion/annotation.pkl
```
We have included the result file by default in the repository. Similarly, we have also included the attribute semantics from GloVe model for all datasets which are computed by:
```
python ./extract_feature/extract_attribute_w2v_DeepFashion.py						        #create ./w2v/DeepFashion_attribute.pkl
python ./extract_feature/extract_attribute_w2v_AWA2.py								#create ./w2v/AWA2_attribute.pkl
python ./extract_feature/extract_attribute_w2v_CUB.py								#create ./w2v/CUB_attribute.pkl
python ./extract_feature/extract_attribute_w2v_SUN.py								#create ./w2v/SUN_attribute.pkl
```

3) Please run feature extraction scripts in `./extract_feature` folder to extract features from the last convolution layers of ResNet as region features for attention mechanism:
```
python ./extract_feature/extract_feature_map_ResNet_101_DeepFashion.py				        #create ./data/DeepFashion/feature_map_ResNet_101_DeepFashion_sep_seen_samples.hdf5
python ./extract_feature/extract_feature_map_ResNet_101_AWA2.py						#create ./data/AWA2/feature_map_ResNet_101_AWA2.hdf5
python ./extract_feature/extract_feature_map_ResNet_101_CUB.py						#create ./data/CUB/feature_map_ResNet_101_CUB.hdf5
python ./extract_feature/extract_feature_map_ResNet_101_SUN.py						#create ./data/SUN/feature_map_ResNet_101_SUN.hdf5
```
These scripts create hdf5 files which contain image features and data splits for training and evaluation.

---
## Training and Evaluation
1) We provide separate jupyter notebooks for training and evaluation on all four datasets in `./notebook`  folder:
```
./notebook/DAZLE_DeepFashion.ipynb
./notebook/DAZLE_AWA2.ipynb
./notebook/DAZLE_CUB.ipynb
./notebook/DAZLE_SUN.ipynb
```

---
## Citation
If this code is helpful for your research, we would appreciate if you cite the work:
```
@article{Huynh-DAZLE:CVPR20,
  author = {D.~Huynh and E.~Elhamifar},
  title = {Fine-Grained Generalized Zero-Shot Learning via Dense Attribute-Based Attention},
  journal = {{IEEE} Conference on Computer Vision and Pattern Recognition},
  year = {2020}}
```
