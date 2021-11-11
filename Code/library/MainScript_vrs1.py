"""
SCRIPT reads .parquet files that contain the power spectra for each workload phase and extracts the features that are the input for the classifiers.
This structure has 2 main steps:
    1. Filter signals using an Inter Quartile Range (IQR filtering). IQR parameters can be computed for each subject or for a training set of 
    2. Cut signals into temporal windows of "window" seconds with an overlapping of "overlap" seconds    
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, plot
from scipy import signal
# from scipy import statscmd
import seaborn as sns
from tqdm import tqdm

from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F


user='AuraCVC'

if user=='Deb':
    proj_dir=r'J:\Experiments\EPilots\ML2\Code\Python\epilots_proj\code\CVCEEGFunctions'
    eeglib = os.path.join(proj_dir,'Networks')
    OutPut_dir=r'J:\Experiments\EPilots\ML2\Results\Deliverable'
    Data_dir=r'J:\Experiments\EPilots\ML2\Data\DualBackTests\input_features'

elif user=='Aura':
    proj_dir = r'G:\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2'
    eeglib = os.path.join(proj_dir,'code/cvc_eeglib')
    dataframes_dir = os.path.join(proj_dir,'dataframes')
    Data_dir= os.path.join(proj_dir,'input_features')
    OutPut_dir = os.path.join(proj_dir,'results')
    
elif user=='AuraCVC':
    proj_dir = r'C:\Users\aura.CVC\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2'
    eeglib = os.path.join(proj_dir,'code/cvc_eeglib')
    dataframes_dir = os.path.join(proj_dir,'dataframes')
    Data_dir= os.path.join(proj_dir,'input_features')
    OutPut_dir = os.path.join(proj_dir,'results')

elif user=='Christian':
    proj_dir = r'D:\UAB\4to\TFG-AI-detection-seizures'
    eeglib = os.path.join(proj_dir,'Data/egg')
    dataframes_dir = os.path.join(proj_dir,'dataframes')
    Data_dir= os.path.join(proj_dir,'Data/parquet')
    OutPut_dir = os.path.join(proj_dir,'results')
    
else:
    proj_dir = r'C:\Users\aura.CVC\OneDrive - Universitat Autònoma de Barcelona\ePilots\ML2'
    eeglib = os.path.join(proj_dir,'code/cvc_eeglib')
    dataframes_dir = os.path.join(proj_dir,'dataframes')
    Data_dir= os.path.join(proj_dir,'input_features')
    OutPut_dir = os.path.join(proj_dir,'results')



sys.path.append(proj_dir)
sys.path.append(eeglib)
sys.path.append(dataframes_dir)
sys.path.append(Data_dir)
sys.path.append(OutPut_dir)

from eeg_globals import *
from signals_functions import *



sns.set(font_scale=1.2, style='white')
#%%
"""
SETTINGS
"""

#dataset on which IQR is applied (options: aslogic,cvc,all,all_plus,selected_dataset,wasim_simulator_dataset,miquel_experiments)
dataset = 'cvc' 
chunk = True
filt = False

dic_filt_opts = {
    'per_phases':False, #can be True and filter each phase by the corresponding phase or False and compute all the signals by a simple IQR computed on a phase
    'datafiltset':'aslogic', #dataset on which IQR is computed. If there is no filtering, change by 'none'
    'setphase': 2, #phase on which IQR is computed 
    'q':0.01, # percentile used to compute IQR
    'IQRtype' : 'new', #'new','old'
    'IQRTh':'iqr'} #qup o iqr


dic_cut = {'window' : 40, # window size seconds
           'overlap' : 30} # overlapping seconds

f_window_name = '_window_' + str(dic_cut['window']) + '_'+ str(dic_cut['overlap'])

if filt:    
    f_filt_name = '_filt_datafiltset_' + dic_filt_opts['datafiltset'] + '_phase_' 
    
    if dic_filt_opts['per_phases']:
        f_filt_name += 'per_phase'
    else:
        f_filt_name += str(dic_filt_opts['setphase']) 
    
    f_filt_name += '_IQR_' + dic_filt_opts['IQRtype']
    if dic_filt_opts['IQRtype'] == 'new':
        f_filt_name += '_' + dic_filt_opts['IQRTh']
else:
    f_filt_name = '_filt_none'
        

# %%
"""
# (0) Extract field data from raw data (for example, power spectra)
# select only the band power channels: theta, alpha, lbeta, hbeta, gamma
filename = dataset + '_eeg.parquet'
eeg_df = pd.read_parquet(os.path.join(dataframes_dir,filename))
print('Step 0. Reduce data, parquet loaded', eeg_df.shape) # (5890910, 142)

pow_nodes= all_pow_nodes #options: pow_theta_nodes, pow_alpha_nodes, pow_betal_nodes, pow_betah_nodes, pow_gamma_nodes, all_pow_nodes
# from globals, chose a list of nodes to be extracted from
selected_cols = qualy_nodes + pow_nodes + user_metalabels # notice that instead of all_pow_nodes you can select an specific wave
eeg_df = eeg_df[selected_cols] # filter out only the the selected columns


eeg_df = eeg_df.dropna() # since rows with NaN values are out the band power band, drop them (276 977, 74)
eeg_df = eeg_df.reset_index(drop=True) # reset index

filename = filename[:-8] + '_power.parquet'
eeg_df.to_parquet(os.path.join(dataframes_dir,filename), engine='pyarrow', compression='brotli', use_deprecated_int96_timestamps=True)
print('reduced parquets saved ', eeg_df.shape) # (368186, 74) 
"""
# %%

# (1) Split data into chunks of N secs
if chunk:

    filename = dataset + '_eeg_power.parquet'
    eeg_df = pd.read_parquet(os.path.join(dataframes_dir,dataset,filename))
    eeg_df['test'] = eeg_df['flight_number']
    print('Step 1. Split data, parquet loaded', eeg_df.shape)    
    if 'simulator' in dataset:
        power_windows_datetime = cut_signal_simulator(eeg_df,dic_cut)
    else:
        power_windows_datetime = cut_signal(eeg_df,dic_cut)
    
    print('saving...')
    
    filename = filename[:-8] + f_window_name + '.parquet'
    filename_d = filename[:-8] + '_datetime.parquet'
    
    ls = list(power_windows_datetime.columns)
    parquet_windows_datetime = power_windows_datetime.take([ls.index('datetime'),ls.index('subject'),ls.index('observation')], axis=1)
    parquet_windows_datetime.to_parquet(os.path.join(dataframes_dir,filename_d), engine='pyarrow', compression='brotli', use_deprecated_int96_timestamps=True)
    eeg_power_window = power_windows_datetime.drop([ 'datetime'], axis=1 )
    eeg_power_window.to_parquet(os.path.join(dataframes_dir,filename), engine='pyarrow', compression='brotli', use_deprecated_int96_timestamps=True)
    print('parquet saved')

    del eeg_power_window,parquet_windows_datetime,power_windows_datetime

# %%
# (2) Evaluate quality of signals


# %%
# (3) Filter the signal by phase

filename = dataset + '_eeg_power' + f_window_name + '.parquet'
eeg_df = pd.read_parquet(os.path.join(dataframes_dir,filename)) 
print('Step 3. Filter signals, parquet loaded', eeg_df.shape)

filename = dataset + '_eeg_power' + f_filt_name + f_window_name + '.parquet'

if filt:
    eeg_pow_filt = filt_signal(proj_dir,eeg_df,dic_filt_opts)
    
    eeg_pow_filt.to_parquet(os.path.join(dataframes_dir,filename), engine='pyarrow', compression='brotli', use_deprecated_int96_timestamps=True)
    del eeg_pow_filt
else:   
    eeg_df.to_parquet(os.path.join(dataframes_dir,filename), engine='pyarrow', compression='brotli', use_deprecated_int96_timestamps=True)


print('parquets saved')




# %%
# (4) Prepare data as rows to feed into the model

# generate two files data_x and data_y

filename = dataset + '_eeg_power' + f_filt_name + f_window_name + '.parquet'
eeg_df = pd.read_parquet(os.path.join(dataframes_dir,filename))
print('Step 4. Create input features, parquet loaded', eeg_df.shape) # (676120, 74)

if 'simulator' in dataset:
    data_x,data_y = input_features_simulator(eeg_df)
else:
    data_x,data_y = input_features_elias(eeg_df)

file_data_x = filename[:-8] + '_data_x.npy'
file_data_y = filename[:-8] + '_data_y.npy'

np.save(os.path.join(Data_dir,file_data_x), data_x)
np.save(os.path.join(Data_dir,file_data_y), data_y)
print('file saved')

# %%
# (5) ML system

"""
llamada a las funciones que entrenan y testean
leave
self
"""
