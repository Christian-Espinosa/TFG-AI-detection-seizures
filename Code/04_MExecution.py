
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
#Based on Check_ModelsArchitecture_Script
from library.CNN_models_vrs1 import *
#from eeg_util_data import * #---> What?



def main():
    #-----------Configure Data Set-----------
    path_chuncker = os.path.abspath("Code\\library\\MainScript_vrs1.py")
    exec(open(path_chuncker).read())

    #-----------Define Hyperparameters-----------
    projmodule_params=None

    convnet_params={}
    convnet_params['kernel_size']=(1,3)
    convnet_params['Nneurons']=16

    outputmodule_params={}
    outputmodule_params['n_classes']=4

    model = CNN_ConcatInput(projmodule_params,convnet_params,outputmodule_params).cuda()

    NSamp=10
    n_channel=14
    L=40
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

if __name__ == '__main__':
    main()