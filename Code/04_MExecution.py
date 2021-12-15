
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

def CheckModel(model):

    print(model)
    #Revisar que el model funciona correctament

    NSamp=1
    n_channel=2
    L=4
    x_test =torch.randn(NSamp,n_channel,L).cuda()
    print("*********************")
    print(x_test)
    print(x_test.shape)
    y = model(x_test)
    print(y.shape)
    print(y)

    NSamp=10
    n_channel=1
    L=40
    x_test=torch.randn(NSamp,n_channel,L)
    model(x_test)

    NSamp=10
    n_channel=14
    L=60
    x_test=torch.randn(NSamp,n_channel,L)
    model(x_test)

#-----------Configure Data Set-----------
#path_chuncker = os.path.abspath("Code\\library\\MainScript_vrs1.py")
#exec(open(path_chuncker).read())

#-------------Open Data-------------
#pathtoChuncks = os.path.abspath("D:\\UAB\\4to\\DataSetTFG\\CVC\\dataframes\\cvc_eeg_power.parquet")
#x_test = pd.read_parquet(pathtoChuncks)

#-----------Define Hyperparameters-----------
projmodule_params=None

convnet_params={}
convnet_params['kernel_size']=(1,3)
convnet_params['Nneurons']=16

outputmodule_params={}
outputmodule_params['n_classes']=4

#CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True



#-----------Train-----------
model = CNN_ConcatInput(projmodule_params,convnet_params,outputmodule_params).cuda()
#CheckModel(model)

#-----------Split DB-----------
labels = ["FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2","FZ-CZ","CZ-PZ","P7-T7","T7-FT9","FT9-FT10","FT10-T8","T8-P8"]

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100


#leave one out -> test con uno
#procentaje dividir
#por cada sujeto trein y test
subj = 'chb01'
parquets = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + '/parquet/')
for name in os.listdir(parquets):
    p_ds = pd.read_parquet(parquets + '/' + name)
    ds_lb = p_ds['labels']
    ds_val = p_ds.drop('labels', axis=1)

    #split dataset amb els labels dintre X Y
    #