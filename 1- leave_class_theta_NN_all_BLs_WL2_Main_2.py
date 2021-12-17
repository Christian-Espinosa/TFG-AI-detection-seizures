# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:55:32 2021

@author: Aura, Jose Elías & Debora
"""

"""
IMPORT LIBRARIES
"""
#%reset -f

# I/O 
import os
import pandas as pd
import sys
import pickle

#NUMERIC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, plot
from scipy import signal
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

# CNNs
import torch
import torch.nn as nn
import torch.nn.functional as F


# Script Own Libraries
user='Aura'

if user=='Deb':
    CodeMainDir=r'J:\Experiments\EPilots\ML2\Code\Python\epilots_proj\code\CVCEEGFunctions'
    eeglib = os.path.join(CodeMainDir,'Networks')
    OutPut_Dir=r'J:\Experiments\EPilots\ML2\Results\Deliverable'
    DataDir=r'J:\Experiments\EPilots\ML2\Data\DualBackTests\input_features'

elif user=='Aura':
    CodeMainDir = r'G:\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2\code'
    eeglib = os.path.join(CodeMainDir,'cvc_eeglib')
    OutPut_Dir=r'G:\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2\input_features'
    DataDir=OutPut_Dir
    
else:
    CodeMainDir = r'G:\OneDrive - Universitat Autònoma de Barcelona\Jose Elias\epilots_proj\code'
    eeglib = os.path.join(CodeMainDir,'eeglib')
    OutPut_Dir=r'G:\OneDrive - Universitat Autònoma de Barcelona\Jose Elias\epilots_proj\data_processed'
    DataDir=OutPut_Dir

sys.path.append(CodeMainDir)
sys.path.append(eeglib)

from eeg_utilitarios import *
from eeg_models import *
from eeg_helpers import *
from eeg_dataset import *
from eeg_globals import *


def TrainModel(x_train,y_train,batch_size,n_epochs,n_classes,lr,x_valid, y_valid):
     # computing a weight per class/sample only is util when
    # you are dealing with unbalanced data, however, it does
    # not matter with balanced dataset

    sample_counts = class_sample_count(list(y_train))
    print('sample_counts: ', sample_counts)
    classes_weight = 1. / torch.tensor(sample_counts, dtype=torch.float)
    samples_weight = torch.tensor([classes_weight[w] for w in y_train])

    print('train shape ', x_train.shape)
    print('valid shape', x_valid.shape)
   
    # Data loader
    train_dataset = EEG_Dataset(x_train, y_train)

    # pytorch function for sampling batch based on weights or probabilities for each
    # element. To obtain a relative balaced batch, it uses replacement by default
    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight,
                                                     num_samples=len(samples_weight),
                                                     replacement=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=False, sampler=sampler)
    
    model = Seq_NN(n_classes=n_classes).cuda()
    # model = Seq_NN(n_classes=n_classes).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    valid_dataset = EEG_Dataset(x_valid, y_valid)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                    shuffle=False)
    model, avg_cost = train_model(model, optimizer, criterion,
                                      train_dataloader,
                                      valid_dataloader,
                                      n_epochs,
                                      verbose=0,
                                      save_path='./pretrained/model.pt')
    #plot_metrics(avg_cost)
    return model

def SelectSubjectTrainTestData(data_x,data_y,subject,test_size):
    # hold-out the chosen subject for test
    chosen_rows = data_y[:, 0] == subject
    train_data_x = data_x[~chosen_rows, :]
    train_data_y = data_y[~chosen_rows, :]
    test_data_x = data_x[chosen_rows, :]
    test_data_y = data_y[chosen_rows, :]
    
    return train_data_x,train_data_y,test_data_x,test_data_y

def FormatData4allBLsModel(train_data_x,train_data_y,channel_id):

    # prepare training set
    # select the wave or frequency of interest in the training set
    x_train = train_data_x[:, channel_id, :] # channel
    y_train = train_data_y[:, -1].astype(np.int)
    
    # keep only BL and WL of interest in the training set
    y_train[ y_train == 11] = 0
    y_train[ y_train == 21] = 0
    y_train[ y_train == 31] = 0
    chosen_rows = (y_train[:] == 0 ) | (y_train[:] == 1 )
    x_train  = x_train[chosen_rows,:]
    y_train  = y_train[chosen_rows]
 
    return x_train, y_train

def ExpDict(class0=['11'],class1=['22'],class2=None):
    
    encoder_dic = {'11' : 11, 
                   '12' : 12,
                   '21' : 21, 
                   '22' : 22, 
                   '31' : 31,
                   '32' : 32}
    
    for c in class0:
        encoder_dic[c] = 0
    for c in class1:
        encoder_dic[c] = 1
    if class2:
        for c in class1:
            encoder_dic[c] = 2
    return encoder_dic   

def prepare_data(filename, class0,class1,class2):
    filename_x = filename + '_data_x.npy' # change the file
    filename_y = filename + '_data_y.npy' 
    
    data_x = np.load(os.path.join(DataDir,filename_x), allow_pickle=True) 
    data_y = np.load(os.path.join(DataDir,filename_y), allow_pickle=True)
    
    
    # get out uninterest tests
    
    data_y = pd.DataFrame(data_y, columns=('subject', 'test', 'phase', 'observation'))
    data_y['label'] = data_y.apply(lambda row: str(row.test) + str(row.phase), axis=1)
    
     
    # goal BL1 - WL2, and others BL2 and BL3, and WL3
    
    
    #eliminem totes les finestres de recovery
    
    data_y = data_y.loc[(data_y.label=='11') | 
                        (data_y.label=='12') | 
                        (data_y.label=='22') | 
                        (data_y.label=='21') | 
                        (data_y.label=='31') |
                        (data_y.label=='32') , :]
    
    # extract data from index
    chosen_rows = list(data_y.index)
    data_x = np.take(data_x, chosen_rows, axis=0)
    data_y = data_y.reset_index(drop=True)
    
    
    # manual label encoder, take care it is the same as above
    # I use a trick during training,  I just will use zero and one
    # however for test I will add other
    encoder_dic=ExpDict(class0,class1,class2)
    data_y['label'] = data_y.apply(lambda row: encoder_dic[row.label], axis=1)
    data_y = data_y.values
       
    return data_x,data_y

#%%
def main(filename,class0,class1,class2=None,class3=None):

    # select the wave here to change all the doc.
    channel_id = 0 # change the wave #necessary?
    
    
    ## CNN Parameters (define hyper-params)
    batch_size = 32
    test_size = 0.10
    
    lr = 1e-4
    n_epochs = 50
    n_exps = 1
    
    """
    LOAD INPUT DATA and prepare their access
    """
    data_x,data_y = prepare_data(filename, class0,class1,class2)
     # taking in account encoder_dic
    if class2:
        if class3:
            tags_categ = ['-'.join(class0),'-'.join(class1),'-'.join(class2),'-'.join(class3)]
            tags_label = [0, 1, 2, 3]
        else:
            tags_categ = ['-'.join(class0),'-'.join(class1),'-'.join(class2)]
            tags_label = [0, 1, 2]        
    else:
        tags_categ = ['-'.join(class0),'-'.join(class1)]
        tags_label = [0, 1]
        
    n_classes = len(tags_categ) # get the number of classes
    subjects = np.unique(data_y[:, 0]) # get the subjects
    
    
    
    # %%
    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    For all subjects
    A summary of model performance for all subjects
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """
    all_scores = []
    all_recalls = []
 
    subject_scores={}  
    
    for subject in subjects:
        
        # Select Subject Data
        train_data_x,train_data_y,test_data_x,test_data_y=SelectSubjectTrainTestData(data_x,data_y,subject,test_size)
    
        # early normalization
        #scalers = {}
        #for i in range(train_data_x.shape[1]):
        #    scalers[i] = preprocessing.StandardScaler()
        #    train_data_x[:, i, :] = scalers[i].fit_transform(train_data_x[:, i, :])
        #for i in range(test_data_x.shape[1]):
        #    test_data_x[:, i, :] = scalers[i].transform(test_data_x[:, i, :])
            
        # Select set 4 assessment of WL3 and BLi separately (others test set)
               
        # Modify Labels for all BLs Model
        x_train0,y_train0=FormatData4allBLsModel(train_data_x,train_data_y,channel_id)
        
        x_train0 = train_data_x[:, channel_id, :]
        y_train0 = train_data_y[:, -1].astype(np.int)
        
        
        # prepare test set
        # select the wave or frequency of interest in the test set
        x_test = test_data_x[:, channel_id, :] #  channel
        y_test = test_data_y[:, -1].astype(np.int)
    
       
        # copy BLs and WL-3 as other data
        chosen_rows = (y_test[:] == 0) | (y_test[:] == 1)
        x_other = x_test[~chosen_rows, :]
        y_other = y_test[~chosen_rows]
    
        # to keep BLs and WL of interest in the training set
        y_test[y_test == 11] = 0
        y_test[y_test == 21] = 0
        y_test[y_test == 31] = 0
        chosen_rows = (y_test[:] == 0) | (y_test[:] == 1)
    
        #overwrite
        x_test = x_test[chosen_rows, :]
        y_test = y_test[chosen_rows]
        
        
        
        chosen_rows = (y_test[:] == 0) | (y_test[:] == 1)
        target_labels=list(np.unique(y_test[chosen_rows].astype(np.int)))  
         
              
        print('test shape ', x_test.shape)
        print('other shape ', x_other.shape)
        np.random.seed(123)
        # WL2 and BLi
        recalls = {}
        precis = {}
        for i in target_labels:
            recalls[i]=[]
            precis[i]=[]
        # WL3 and BLj, j diferent de i
        recalls_others ={}
        
        other_labels = list(np.unique(y_other))
        for i in other_labels:
            recalls_others[i]=[]
         
        """
        COMPUTE K-FOLD TRAINING FOR EACH SUBJECT
        """   
        for exp in range(n_exps): # n experiments
                   
            # split training data into train and valid sets
            x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
                            x_train0, y_train0,
                            test_size=test_size,
                            stratify=y_train0)
            # 2. Normalize Data
            # normalize train data
            # source https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
            
            scalers = {}
            for i in range(x_train.shape[1]):
                scalers[i] = preprocessing.StandardScaler() 
                x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :])
        
            # normalize valid data
            for i in range(x_valid.shape[1]):
                x_valid[:, i, :] = scalers[i].transform(x_valid[:, i, :])
        
             # normalize test data
            for i in range(x_test.shape[1]):
                x_test[:, i, :] = scalers[i].transform(x_test[:, i, :])
                
             # normalize other data
            for i in range(x_other.shape[1]):
                x_other[:, i, :] = scalers[i].transform(x_other[:, i, :])
        
            # 3. Train Models
            print('train shape ', x_train.shape)
            torch.manual_seed(123)
            model=TrainModel(x_train,y_train,batch_size,n_epochs,n_classes,lr,x_valid, y_valid)
    
            # 4. Test Model
            # 4.1 Test on WL2, BLi used for training model
        
            test_dataset = EEG_Dataset(x_test, y_test)
            test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False)
        
            _,_, (y_true, y_pred) = test_model(model, test_dataloader)
            report = metrics.classification_report(y_true, y_pred, labels=tags_label,target_names=tags_categ, zero_division=0, output_dict=True)
            
            for i in np.arange(len(target_labels)):
                current_lab=target_labels[i]
                recalls[current_lab].append(np.rint(report[tags_categ[i]]['recall'] * 100).astype('int'))
                precis[current_lab].append(np.rint(report[tags_categ[i]]['precision'] * 100).astype('int'))
            
        
            #4.2 Test on WL3, BLj, j diferent de i
            for i in other_labels:
                recalls_others[i]
                msg = str(i)
                if msg.endswith('1'):
                     id_class = 0
                 #   id_task = dic_baseline[i]
                else:
                    id_class = 1
                 #   id_task = 'WL-3'
                
                chosen_rows = (y_other == i)
                # create dataloaders for other data
                other_dataset = EEG_Dataset(x_other[chosen_rows, :], y_other[chosen_rows])
                other_dataloader = torch.utils.data.DataLoader(other_dataset,
                            batch_size=len(other_dataset),
                            shuffle=False)
                y_true = [id_class] * len(other_dataset) # make all labels to first class label
            
                seqs, _ = next(iter(other_dataloader))
        
                outputs = model(seqs.cuda())
                y_pred = outputs.max(1)[1].cpu().numpy()
                # to recovery recall
                report = metrics.classification_report(y_true, y_pred, labels=tags_label,target_names=tags_categ, zero_division=0, output_dict=True)
                recalls_others[i].append(np.rint(report[tags_categ[id_class]]['recall'] * 100).astype('int'))    
            ### End nTrials 4 statistics    
        subject_scores[subject] = {'recalls_target':recalls, 'precis_target':precis ,'recalls_others':recalls_others}
        
    ### End Subject loop
        
    """
    SAVE EXPERIMENT RESULS
    """
    ExpOutPut = {}
    ExpOutPut['subject_scores']=subject_scores
    ExpOutPut['other_labels']=other_labels
    ExpOutPut['target_labels']=target_labels
    
    ReFile='leave_' + filename + '_NN_class0' + str(class0) + '_class1'+str(class1)+ '_class2'+str(class2)+'.pickle'
    ResFile=os.path.join(OutPut_Dir,ReFile)
    with open(ResFile,'wb') as outfile:
        pickle.dump(ExpOutPut,outfile)



"""
MAIN SCRIPT
"""
class0 = ['11','21','31']
class1 = ['22']
filename = 'selected_dataset_eeg_power_filt_datafiltset_aslogic_phase_2_IQR_new_iqr_window_5_0' # change the file
print('1st',class0,class1)
main(filename,class0,class1)

