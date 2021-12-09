
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-----------Train-----------
model = CNN_ConcatInput(projmodule_params,convnet_params,outputmodule_params).cuda()
CheckModel(model)

#-----------Split DB-----------
#leave one out -> test con uno
#procentaje dividir
#por cada sujeto trein y test
