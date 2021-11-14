# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:01:32 2021

@author: debora
"""
import torch
import torch.nn as nn

import numpy as np

def class_sample_count(labels):
    tags = set(labels) # unique categories
    my_dic = {i:labels.count(i) for i in tags}
    my_dic = dict(sorted(my_dic.items())) # weight should be ordered for the optimizer
    print(my_dic)
    samples = list(my_dic.values())
    return samples

def WeightedCrossEntropy(y_train):
    
    
    # computing a weight per class/sample only is util when
    # you are dealing with unbalanced data, however, it does
    # not matter with balanced dataset
    sample_counts = np.array(class_sample_count(list(y_train)))
    classes_weight=1./sample_counts
    classes_weight=classes_weight/np.sum(classes_weight)
    classes_weight=torch.tensor(classes_weight, dtype=torch.float).cuda()
    criterion = nn.CrossEntropyLoss(weight=classes_weight)
        
    return criterion,classes_weight
