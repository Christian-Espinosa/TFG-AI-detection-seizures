import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
#import torchvision

import numpy as np

import matplotlib.pyplot as plt

# =============================================================================
# Standard dataset 
# =============================================================================
# X needs to have structure [NSamp,...] or be a list of NSamp entries
class Standard_Dataset(data.Dataset):
    def __init__(self, X, Y, transformation=None):
        super().__init__()
        self.X = X
        self.y = Y
        self.transformation = transformation

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(np.array(self.y[idx])).long()

# =============================================================================
# Transformations
# =============================================================================

class IdentitySeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):
        return x  # [n_feats, time_steps]

class ReverseSeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):
        x = np.ascontiguousarray(x[:, ::-1])
        return x # [n_feats, time_steps]

class ToTensorSeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):

        return torch.from_numpy(x) # [n_feats, time_steps]

"""
TFG:Valorar si val la pena posar això en un .py a part
    
"""
# =============================================================================
# Data Sets Creation
# =============================================================================

def create_dataloader(x_test, y_test, transf=False, batch_size=128, shuffle=False):
    if y_test.ndim > 1:
        print('Data was no properly encoded. Error in test_dataloader!')
        return

    if y_test.shape[0] < batch_size:
        batch_size = y_test.shape[0]

    test_dataset = Standard_Dataset(x_test, y_test, transf)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=shuffle)
    return test_dataloader


def create_dataloader_balanced(x_train, y_train, transf=False, batch_size=128, shuffle=False):
    if y_train.ndim > 1:
        print('Data was no properly encoded. Error in train_dataloader!')
        return

    sample_counts = class_sample_count(list(y_train))
    classes_weight = 1. / torch.tensor(sample_counts, dtype=torch.float)
    samples_weight = torch.tensor([classes_weight[w] for w in y_train])

    # traind dataloader
    train_dataset = Standard_Dataset(x_train, y_train, transf)
    print('training set ', x_train.shape[0])

    # pytorch function for sampling batch based on weights or probabilities for each
    # element. To obtain a relative balaced batch, it uses replacement by default
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=shuffle, sampler=sampler)
    return train_dataloader

# =============================================================================
#
# =============================================================================
def class_sample_count(labels):
    tags = set(labels) # unique categories
    my_dic = {i:labels.count(i) for i in tags}
    my_dic = dict(sorted(my_dic.items())) # weight should be ordered for the optimizer
    print(my_dic)
    samples = list(my_dic.values())
    return samples

def class_weight(labels):
    tags = set(labels) # unique categories
    my_dic = {i:labels.count(i) for i in tags}
    my_dic = dict(sorted(my_dic.items())) # weight should be ordered for the optimizer
    print(my_dic)
    max_val = max(my_dic.values())
    weights = [ round(max_val / my_dic[i], 2)  for i in my_dic.keys()]
    #weights = [ 1. / my_dic[i]  for i in my_dic.keys()]
    return weights