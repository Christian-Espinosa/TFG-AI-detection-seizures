
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
import library.Dataset_Functions as dat



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



# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6,
          'n_epochs': 100}
criterion = nn.CrossEntropyLoss()
model = CNN_ConcatInput(projmodule_params,convnet_params,outputmodule_params)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, momentum = 0.9)
subj = 'chb01'

#dat.CheckModel(model)

parquets = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + '/parquet/')
for name in os.listdir(parquets):
    p_ds = pd.read_parquet(parquets + '/' + name)

    train, test =  dat.SplitData(p_ds, 0.01)
    #Validation?

    #Train
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"], shuffle=params["shuffle"])

    model, avg_cost = dat.train_model(model,
                                    optimizer,
                                    criterion,
                                    train_dataloader,
                                    valid_dataloader = None,
                                    batch_size=params["n_epochs"],
                                    verbose=0,
                                    save_path= parquets + '/model.pt')

    
    #Test
    test_dataloader = torch.utils.data.DataLoader(test.loc[:, train.columns != 'b'], batch_size=params["batch_size"], shuffle=params["shuffle"])