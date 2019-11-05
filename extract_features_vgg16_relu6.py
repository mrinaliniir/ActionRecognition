#!/usr/bin/env python
# coding: utf-8

# # Action Recognition @ UCF101  
# **Due date: 11:59 pm on Dec. 11, 2018 (Tuesday)**
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, utils
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
import scipy.io 
from time import time
import pickle


# In[6]:


vgg16 = models.vgg16(pretrained=True)

new_classifier = nn.Sequential(*list(vgg16.classifier.children())[:1])
vgg16.classifier = new_classifier


# In[6]:


# write your codes here
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

prep = transforms.Compose([ transforms.ToTensor(), normalize ])

data = []
labels = []

t0 = time()
for filename in sorted(os.listdir('UCF101_release/images_class1/')):
    image_feature = np.empty(shape=[0,4096])
    for f in sorted(os.listdir('UCF101_release/images_class1/'+filename)):
        input_tensor = []
        img = cv2.imread('UCF101_release/images_class1/'+filename+"/"+f)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #left up
        left_up = img[0:224,0:224]
        left_up = prep(left_up)

        #right up
        right_up = img[0:224,116:116+224]
        right_up = prep(right_up)

        #center
        center = img[16:16+224,58:58+224]
        center = prep(center)

        #left_down
        left_down = img[32:32+224,0:224]
        left_down = prep(left_down)

        #right_down
        right_down = img[32:32+224,116:116+224]
        right_down = prep(right_down)
        
        features = vgg16(torch.stack([left_up,right_up,center,left_down,right_down]))
        output_features = torch.sum(features,dim = 0)   #return size [1,4096]
        output_features = output_features*1.0 / 5
        image_feature = np.vstack((image_feature,output_features.detach().numpy()))

    print(image_feature.shape)
    scipy.io.savemat('UCF101_release/vgg16_relu6/'+filename,{'Feature':image_feature},do_compression = True)
    print("Writing to "+'UCF101_release/vgg16_relu6/'+filename+'.mat')
    
print('Feature Extraction complete in {:.0f}m {:.0f}s'.format((time()-t0) // 60, (time()-t0) % 60))
