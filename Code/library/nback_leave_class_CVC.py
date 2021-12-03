"""
    Documentation: Elias        

        In the report:
            class 0, without interruptions
            class 1, with interruptions
"""
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
user='AuraCVC'

if user=='Deb':
    CodeMainDirDeb=r'J:\Experiments\EPilots\ML2\Code\Python\epilots_proj\code\CVCEEGFunctions'
    CodeMainDirElias=r'J:\Experiments\EPilots\ML2\Code\Python\epilots_proj\code\EliasFunctions'
    EliaslibDirs=['cvc_eeglib']

    sys.path.append(CodeMainDirElias)
    for lib in EliaslibDirs:
        sys.path.append(os.path.join(CodeMainDirElias,lib))
    sys.path.append(CodeMainDirDeb)


    OutPut_Dir=r'J:\Experiments\EPilots\ML2\Results\SeriousGame'
    DataDir=r'Y:\Database\Cognitive\Sensors\WorkLoad\PrivateBD\E-Pilots_Aslogic\SeriousGame'

elif user=='Aura':
    CodeMainDir = r'G:\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2\code'
    CodeMainDirElias=r'G:\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2\code\EliasFunctions'
    eeglib = os.path.join(CodeMainDir,'cvc_eeglib')
    DataDir=r'G:\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2\input_features\selected'
    OutPut_Dir = r'G:\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2\results\mdpi'
    EliaslibDirs=['cvc_eeglib']

    sys.path.append(CodeMainDirElias)
    for lib in EliaslibDirs:
        sys.path.append(os.path.join(CodeMainDirElias,lib))
    sys.path.append(CodeMainDir)
elif user=='AuraCVC':
    CodeMainDir = r'C:\Users\aura.CVC\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2\code'
    CodeMainDirElias=r'C:\Users\aura.CVC\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2\code\EliasFunctions'
    eeglib = os.path.join(CodeMainDir,'cvc_eeglib')
    DataDir=r'C:\Users\aura.CVC\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2\input_features\selected'
    OutPut_Dir = r'C:\Users\aura.CVC\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2\results\mdpi'
    EliaslibDirs=['cvc_eeglib']

    sys.path.append(CodeMainDirElias)
    for lib in EliaslibDirs:
        sys.path.append(os.path.join(CodeMainDirElias,lib))
    sys.path.append(CodeMainDir)
else:
    
    CodeMainDir = r'D:/project_pytorch/ex_eeg'
    eeglib = os.path.join(CodeMainDir,'cvc_eeglib')    
    DataDir = r'\\iamdata\IAM\Database\Cognitive\Sensors\WorkLoad\PrivateBD\E-Pilots_Aslogic\SeriousGame'
    OutPut_Dir = r'D:\outputs'
    sys.path.append(CodeMainDir)
    sys.path.append(eeglib)



import eeg_globals as gbl
from .eeg_util_data import *
from .eeg_util_models import *
from .eeg_util_plots import *
from .eeg_util_conf_mat import *

#%%
def main(filename, MODEL_NAME, TRAIN_CONFIG, EXP_TYPE, SAVE_PTH):

    WIN_SIZE = int(filename.split('_')[-2])

    data_x, data_y = read_input_features(DataDir, filename)
    
    data_x = data_x[:, TRAIN_CONFIG['channel_id'], :, :]     # take the wave of interest


    # data_x, data_y = seriousgame_select_data_Deb(data_x, data_y,EXP_TYPE) 
    data_x, data_y = select_data_Aura(data_x, data_y,EXP_TYPE) 
    # data_x_ts, data_y_ts = seriousgame_select_data_Deb(data_x, data_y,'Test') 


    # taking in account encoder_dic
    tags_categ = EXP_TYPE.split('_')
    tags_label = np.arange(len(tags_categ)) # generate codes
    n_classes = len(tags_categ) # get the number of classes
    subjects = np.unique(data_y[:, 0]) # get the subjects
    
    
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    For all subjects
    A summary of model performance for all subjects
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    subject_scores={}
    subject_model={}
    for subject in subjects:
    # for subject in ['agusti']: # ELIAS

        print('Testing ', subject)
        np.random.seed(123)

        # select the subject data to be tested
        train_data_x, train_data_y, test_data_x, test_data_y = select_subject_train_test_data(data_x, data_y, subject)
   #     _, _, test_data_x, test_data_y = select_subject_train_test_data(data_x_ts, data_y_ts, subject)

        # 1) data normalization
        train_data_x, scalers = scalers_fit(train_data_x)
        test_data_x = scalers_transform(scalers, test_data_x)

        # 2) prepare train set
        x_train0, y_train0 = train_data_x, train_data_y[:,-1].astype(np.int64)

        # 3) prepare test set
        x_test, y_test = test_data_x, test_data_y[:,-1].astype(np.int64)
    #    print(subject, ' -> ', x_test.shape[0])

        target_labels = sorted(list(np.unique(y_test)))

        # WLi and BLi
        recalls = {}
        precis = {}
        for i in target_labels:
            recalls[i]=[]
            precis[i]=[]

        
        """
        4) COMPUTE K-FOLD TRAINING FOR EACH SUBJECT
        """
        for exp in range(TRAIN_CONFIG['n_exps']): # n experiments

            # a. split training data into train and valid sets
            if TRAIN_CONFIG['test_size'] <= 0 :
                x_train = x_train0
                y_train = y_train0
                x_valid = None
                y_valid = None
            else:

                x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
                            x_train0, y_train0,
                            test_size=TRAIN_CONFIG['test_size'],                            
                            stratify=y_train0, random_state=123)

            # b. train model
            torch.manual_seed(123) # for reproducibility            
            # se podria colocar pth_full_name si deseamos guardar en cada epoca
            model, optimizer, avg_cost = train_Deb(x_train, y_train, 
                                           n_classes, 
                                           TRAIN_CONFIG['batch_size'], 
                                           TRAIN_CONFIG['n_epochs'],
                                           TRAIN_CONFIG['lr'], 
                                           TRAIN_CONFIG['step_size'], 
                                           TRAIN_CONFIG['transf'], 
                                           x_valid, y_valid, 
                                           MODEL_NAME, 
                                           WIN_SIZE, pth_full_name=None)


            # save the model by subject at the end of training
            if SAVE_PTH:

                pth_dir = os.path.join(OutPut_Dir, 'pretrained', MODEL_NAME)
                if not os.path.exists(pth_dir):
                    os.makedirs(pth_dir)

                pth_file =  subject + '_md_' + MODEL_NAME + '_ep_' + str(TRAIN_CONFIG['n_epochs']) + \
                        '_lr_' + '{:.0e}'.format(TRAIN_CONFIG['lr']) + '_exp_' + EXP_TYPE + '.pth'
                pth_file = os.path.join(pth_dir, pth_file)
                save_checkpoint(model, optimizer, np.Inf, TRAIN_CONFIG['n_epochs'], pth_file)

            # to plot training loss
    #        plot_metrics(avg_cost, msg=subject)

            # c. test model
            test_dataset = EEG_Dataset(x_test, y_test)
            test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                       batch_size=TRAIN_CONFIG['batch_size'],
                                                       shuffle=False)

            y_true, y_pred, y_prob = test_model(model, test_dataloader)
            report = metrics.classification_report(y_true, y_pred, labels=tags_label,
                                                   target_names=tags_categ,
                                                   zero_division=0, output_dict=True)

            # to print metrics
            # print(metrics.confusion_matrix(y_true, y_pred))
            # print(metrics.classification_report(y_true, y_pred, labels=tags_label,target_names=tags_categ, zero_division=0))

            # to plot confusion matrix
          #  confusion_matrix_plot(y_true, y_pred, title=subject + ' <-> ' + MODEL_NAME, tags_categ=tags_categ)

            # to save confusion matrix
            c_m = confusion_matrix_calculate(y_true, y_pred, tags_categ)

            # to print train config
    #        print(EXP_TYPE, ' -> epochs ', TRAIN_CONFIG['n_epochs'],' | lr ', '{:.0e}'.format(TRAIN_CONFIG['lr']),' | win ', WIN_SIZE)

            for i in target_labels:
                recalls[i].append(np.rint(report[tags_categ[i]]['recall'] * 100).astype(np.int32))
                precis[i].append(np.rint(report[tags_categ[i]]['precision'] * 100).astype(np.int32))
            
            ### End nTrials 4 statistics
        subject_scores[subject] = {'recalls_target':recalls,
                                   'precis_target':precis ,
                                   'confusion_matrix' : c_m,
                                   'probabilities': y_prob}
        subject_model[subject]={'model': model,
                                'scalers': scalers,
                                'avg_cost': avg_cost}

    ### End Subject loop

    """
    SAVE EXPERIMENT RESULS
    """
    ExpOutPut = {}
    ExpOutPut['subject_scores'] = subject_scores    
    ExpOutPut['target_labels'] = target_labels
    ExpOutPut['metadata'] = TRAIN_CONFIG
    ExpOutPut['subject_model']=subject_model

    ReFile =    'leave_' + filename + '_md_' + MODEL_NAME + '_ch_' + str(TRAIN_CONFIG['channel_id']) + \
                '_ep_' + str(TRAIN_CONFIG['n_epochs']) + '_lr_' + '{:.0e}'.format(TRAIN_CONFIG['lr']) + \
                '_exp_' + EXP_TYPE + '.pickle'

    ResFile=os.path.join(OutPut_Dir, ReFile)
    with open(ResFile,'wb') as outfile:
        pickle.dump(ExpOutPut, outfile)
    outfile.close()
# %%

"""
MAIN SCRIPT
"""

#if __name__ == '__main__':
# 0 TITAN      12 GB
# 1 SUPER       8  GB
torch.cuda.set_device(0)
id_cuda = torch.cuda.current_device()
#    print('working on: ', torch.cuda.get_device_name(id_cuda))

SAVE_PTH = False
#
EXP_TYPE = 'BLs_WL2'
# EXP_TYPE = 'WL1_WL2_WL3'




TRAIN_CONFIG = {
    'n_exps'        : 1,    # internally, number of running experiments
    
    'transf'        : False, # tranformation data
       
    'batch_size'    : 750,  # batch_size according to GPU memory available

    'test_size'     : 0.05, # test size, if zero, NO validation set
    
    'step_size'     : 0,    # epoch to change learning rate, if ZERO, None

    'n_epochs'      : 100,  # number of epochs

    'lr'            : 1e-4, # starting learning rate

    'channel_id'    : 0,    # wave                        
}


MODEL_NAMEs = ['CNN']    # 'C2DSpatTem1', 'C2DSpatTem2'

# EXP_TYPEs = ['WL1_WL2_WL3','BLs_WL2_WL3','BLs_WL1_WL2_WL3'] #'BLs_WL2',

# r'selected_dataset_eeg_power_filt_datafiltset_aslogic_phase_2_IQR_new_iqr_window_5_0',
# r'selected_dataset_eeg_power_filt_datafiltset_aslogic_phase_2_IQR_new_iqr_window_10_5',
#             r'selected_dataset_eeg_power_filt_datafiltset_aslogic_phase_2_IQR_new_iqr_window_20_10',
files = [ 
             r'cvc_eeg_power_filt_none_window_40_30',
        ]
channel_id = 0

OutPut_Dir += '/selected_' + EXP_TYPE
if not os.path.exists(OutPut_Dir):
    os.makedirs(OutPut_Dir)
for filename in files:
    for MODEL_NAME in MODEL_NAMEs:
        # for channel_id in np.arange(5):
            TRAIN_CONFIG['channel_id']=channel_id
            print('experiment on file: ', filename)
            main(filename, MODEL_NAME, TRAIN_CONFIG, EXP_TYPE, SAVE_PTH)
            plt.close('all')
        