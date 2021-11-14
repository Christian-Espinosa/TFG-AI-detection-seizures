"""
    Documentation: Elias        

        In the report:
            class 0, without interruptions
            class 1, with interruptions
"""
## 0. INPUT PACKAGES
# I/O
import os
import pandas as pd
import sys
import pickle
import glob

#NUMERIC
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

# CNNs
import torch
import torch.nn as nn
import torch.nn.functional as F



# Script Own Libraries
user='Deb'

if user=='Deb':
    CodeMainDir=r'J:\Funcions_Python\Networks'

    OutPut_Dir=r''
    DataDir=r' '

else:
    CodeMainDir=r''

    OutPut_Dir=r''
    DataDir=r' '

# Add Code dirs to path
libDirs=next(os.walk(CodeMainDir))[1]
sys.path.append(CodeMainDir)
for lib in libDirs:
    sys.path.append(os.path.join(CodeMainDir,lib))


from CNN_models_vrs1 import *
from synthetic_datasets import *
from train_models import *

"""
TFG: Replicate for:
    1. All architectures defined in CNN_models_vrs1
    2. CNN_models re-implemented to support flexible blocks. In this case start with 
    the same architecture of blocks that was already defined in CNN_models_vrs1 (ConVNet of 3 blocks of 1 layer each)
    and then try with different number of blocks and layers per block
"""

## 1. MODEL DEFINITION
# Model Parameters
convnet_params={}
#convnet_params['kernel_size']=[ ]
#convnet_params['kernel_size'].append([(1,3)])
convnet_params['kernel_size']=(1,3)
convnet_params['Nneurons']=16
projmodule_params=None
outputmodule_params={}
outputmodule_params['n_classes']=4
MODEL_CONFIG={}
MODEL_CONFIG['convnet_params']=convnet_params
MODEL_CONFIG['projmodule_params']=projmodule_params
MODEL_CONFIG['outputmodule_params']=outputmodule_params
MODEL_CONFIG['MODEL_NAME']='CNN_ConcatInput'
model, optimizer, scheduler=instance_model(MODEL_CONFIG,OPTIMIZER_CONFIG)

## 2. DATA SET DEFINITION
NSamp=400

# DataSet Parameters
data_params={}
data_params['n_channel']=[14,1,14]
data_params['L']=[40,40,60]

NSamp=400
n_channel=14
L=40

data_x,data_y=ConstantDataSet(outputlayer_params['n_classes'],L,NSamp)

## 3. MODEL TRAINING
# Parameters
TRAIN_CONFIG = {
        
    'transf'        : False, # tranformation data
       
    'batch_size'    : 500,  # batch_size according to GPU memory available

    'test_size'     : 0.05, # test size, if zero, NO validation set
    
    'n_epochs'      : 70,  # number of epochs
    'balanced'      : False, #whether train set should be balanced (unnecessary in case weighted loss is used) 
    'pth_full_name' : None #Folder where save checkpoints
            
}


OPTIMIZER_CONFIG={}
OPTIMIZER_CONFIG['lr']=1e-4, # starting learning rate
OPTIMIZER_CONFIG['step_size']=30 # epoch to change learning rate, if ZERO, None

MODEL_CONFIG['model']=model


# Folds Definition
NFold=30
kf =  StratifiedKFold(n_splits=NFold)

## 4. MODEL TESTING
# Evaluate Model Architecture

x_test=torch.randn(NSamp,n_channel,L)
model(x_test.cuda())

NSamp=10
n_channel=1
L=40
x_test=torch.randn(NSamp,n_channel,L)
model(x_test.cuda())

NSamp=10
n_channel=14
L=60
x_test=torch.randn(NSamp,n_channel,L)
model(x_test.cuda())
