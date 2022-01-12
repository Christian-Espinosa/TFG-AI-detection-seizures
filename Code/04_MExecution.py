
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
from library.CNN_models_vrs2 import *
#from eeg_util_data import * #---> What?
import library.Dataset_Functions as dat

from  library.eeg_util_data import *

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
convnet_params['Nneurons']=22

outputmodule_params={}
outputmodule_params['n_classes']=2

#CUDA
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
device = torch.device("cpu")
torch.backends.cudnn.benchmark = True



# Parameters
params = {'batch_size': 750,
          'shuffle': False,
          'num_workers': 6,
          'n_epochs': 100}
criterion = nn.CrossEntropyLoss()
model = CNN_ConcatInput(projmodule_params,convnet_params,outputmodule_params)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
subj = 'chb01'

dat.CheckModel(model, device)
#Mirar de guardar el modelo
#torch.save(model)
"""
parquets = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + '/numpy/')
save_model = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj)
for name in os.listdir(parquets):
    p_ds = pd.read_parquet(parquets + '/' + name)
    print(p_ds[0:30 ])
    # 1) data normalization
    data, scalers = scalers_fit(p_ds)
    data = scalers_transform(scalers, data)

    train, test =  dat.SplitData(p_ds, 0.01)
    #Validation?

    #Train
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"], shuffle=params["shuffle"])
    model, avg_cost = dat.train_model(model, optimizer, criterion, train_dataloader)


    #Test
    test_dataloader = torch.utils.data.DataLoader(test.loc[:, train.columns != 'b'], batch_size=params["batch_size"], shuffle=params["shuffle"])

    y_true, y_pred, y_prob = dat.test_model(model, test_dataloader)
    report = metrics.classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    print(report)
    
    """
"""
if save_model:

    pth_dir = os.path.join(OutPut_Dir, 'pretrained', MODEL_NAME)
    if not os.path.exists(pth_dir):
        os.makedirs(pth_dir)

    pth_file =  subject + '_md_' + MODEL_NAME + '_ep_' + str(TRAIN_CONFIG['n_epochs']) + \
            '_lr_' + '{:.0e}'.format(TRAIN_CONFIG['lr']) + '_exp_' + EXP_TYPE + '.pth'
    pth_file = os.path.join(pth_dir, pth_file)
    save_checkpoint(model, optimizer, np.Inf, TRAIN_CONFIG['n_epochs'], pth_file)
    """
    
    