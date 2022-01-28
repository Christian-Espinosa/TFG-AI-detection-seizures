
## 0. INPUT PACKAGES
# I/O
import os
import pandas as pd
import sys
import pickle
import glob
from datetime import datetime

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
from library.eeg_dataset import *
from  library.eeg_util_data import *

#-----------Configure Data Set-----------
path_model = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/trys")
path_load_model = os.path.abspath(path_model + "/" + ""+ ".pt")


#-----------Define Hyperparameters-----------
projmodule_params=None
convnet_params={}
convnet_params['kernel_size']=(1,3)
convnet_params['Nneurons']=22
outputmodule_params={}
outputmodule_params['n_classes']=2

#CUDA
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
#device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 750,
          'shuffle': False,
          'num_workers': 6,
          'n_epochs': 1}#100
criterion = nn.CrossEntropyLoss()
model = CNN_ConcatInput(projmodule_params,convnet_params,outputmodule_params)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

save_model = True

#check model?
if False:
    dat.CheckModel(model, device)
else:

    #Train only/Test only
    if True:

        #Subjects
        n_subjects = 1
        for i in range(1,n_subjects+1):
            subj = 'chb' + "{:02.0f}".format(i)
            print("Subject: ", subj)
            numpys = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + '/numpy/')
            
            #Numpys
            npys = os.listdir(numpys)
            for file in range(0,int(len(npys)),2):

                print("Loading:".format(npys[file]))
                data_x = np.load(os.path.join(numpys,npys[file]), allow_pickle=True)
                data_y = np.load(os.path.join(numpys,npys[file+1]), allow_pickle=True)
                
                train_data_x, train_data_y, test_data_x, test_data_y = dat.select_subject_train_test_data(data_x, data_y, 0.99)
                print("data loaded and splitted")

                # 1) Data normalization
                train_data_x, scalers = dat.scalers_fit(train_data_x)
                test_data_x = dat.scalers_transform(scalers, test_data_x)

                print("start Train...")
                # 2) Train
                train_dataset = EEG_Dataset(train_data_x, train_data_y)
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=params["shuffle"])
                model, avg_cost = dat.train_model(device, model, optimizer, criterion, train_dataloader, None, params["n_epochs"])

                print("Done!")
                # 3) Save model
            if save_model:
                torch.save(model, os.path.abspath(path_model + "/" + datetime.now().strftime("%d%m%Y_%H%M%S_") + subj + ".pt"))
                print("Model Saved!")

    else:
        
        torch.load(os.path.abspath(path_model + "/" + ".pt"))

        for file in range(1,len(os.listdir(numpys)/2)):
            #Test
            test_dataset = EEG_Dataset(test_data_x, test_data_y)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=params["shuffle"])

            y_true, y_pred, y_prob = dat.test_model(device, model, test_dataloader)
            
            #y_true ->  grountruth 0, 1 si es seizure o no
            #y_perd -> 0, 1 segun haya predicho el modelo
            #y_prob ->  probabilidades 
        
        report = metrics.classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        print(report)
        
            