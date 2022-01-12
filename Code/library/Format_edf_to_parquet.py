from os import path
import os, sys
import pyedflib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from scipy.signal import butter, lfilter

import warnings
warnings.filterwarnings("ignore")

def edf_to_numpy(name_edf):
    n = name_edf.signals_in_file
    signal_labels = name_edf.getSignalLabels()
    sigbufs = np.zeros((n, name_edf.getNSamples()[0]))

    for i in np.arange(n):
        sigbufs[i, :] = name_edf.readSignal(i)

    return signal_labels, sigbufs

def Create_parquet_from_edf(name_edf, path_parquet=None):
    n = name_edf.signals_in_file
    signal_labels = name_edf.getSignalLabels()
    sigbufs = np.zeros((n, name_edf.getNSamples()[0]))

    for i in np.arange(n):
        sigbufs[i, :] = name_edf.readSignal(i)
    
    dic = dict(zip(signal_labels, sigbufs))
    df = pd.DataFrame(list(dic.items()), columns=['Electrode', 'Value'])
    if path_parquet!=None:
        pq.write_table(pa.Table.from_pandas(df), path_parquet)

    #Returns pandas table
    return df, dic

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
"""
def set_seizure_labeling(df, edf_f, path_parquet=None):
    #Data file
    n = edf_f.signals_in_file
    signal_labels = edf_f.getSignalLabels()
    sigbufs = np.zeros((n, edf_f.getNSamples()[0]))

    for i in np.arange(n):
        sigbufs[i, :] = edf_f.readSignal(i)

    
    df.append('seizure', sigbufs)
    if path_parquet!=None:
        pq.write_table(pa.Table.from_pandas(df), path_parquet)

    #Returns pandas table
    return df
"""
def setLabels(dic, f, n, hz=256):
    #sets labels to a file using summary
    #dic --> dictionary of the file
    #f --> path to summary of the file
    #hz --> hertz of samples

    dic['seizure'] = np.zeros(len(dic[list(dic.keys())[-1]]), dtype=np.int8)
    name = f[-17:][:-12] #Get last part of string and delete .txt
    with open(f, 'r') as file:
        lines = file.readlines()
    for i in range(len(lines)):
        if lines[i].strip() == "File Name: " + name + "_" + "{:02.0f}".format(n) + ".edf":
            i += 3
            if lines[i].strip() == "Number of Seizures in File: 0":
                return dic
            elif lines[i].strip()[:-1] == "Number of Seizures in File: ":
                for _ in range(int(lines[i].strip()[-1])):
                    i += 1
                    ini = int(lines[i].strip()[:-8].replace('Seizure Start Time: ',''))*hz
                    i += 1
                    fi = int(lines[i].strip()[:-8].replace('Seizure End Time: ',''))*hz
                    for x in range(ini, fi):
                        dic['seizure'][x] = 1
                return dic
    return None

def saveToParquet(dic, path):
    pd.DataFrame.from_dict(dic).to_parquet(path)

def chunkData(path, dic_cut):
    
    eeg_df = pd.read_parquet(os.path.abspath(path))
    print('Step 1. Split data, parquet loaded', eeg_df.shape)
    print('Cutting signal...')
    df_windows = cut_signal_CHB(eeg_df,dic_cut)
    print('saving...')
    df_windows.to_parquet(path, engine='pyarrow', compression='brotli', use_deprecated_int96_timestamps=True)
    print('parquet saved')

    del df_windows

def saveToNumpy(path_parquet, path_numpy):
    # generate two files data_x and data_y
    eeg_df = pd.read_parquet(path_parquet)
    print('Generate two files data_x and data_y. Create input features, parquet loaded', eeg_df.shape)
    
    print('Creating numpy xy...')
    data_x, data_y = input_features(eeg_df)

    print("Saving numpys...")
    np.save(os.path.abspath(path_numpy + '_data_x.npy'), data_x)
    np.save(os.path.abspath(path_numpy + '_data_y.npy'), data_y)
    print('file saved')
    
def cut_signal_CHB(df,dic_cut, hz=256):
    
    df['observation'] = -1
    
    #Function to split data into windows to feed the model
    sample_win = int(hz * dic_cut['window'])
    sample_over = int(hz * dic_cut['overlap'])
    sample_stride = sample_win - sample_over
    
    
    # To data, add a column observation based on phase
    print('Split data into windows ("observation" label)')
    n_intervals = int(np.floor(( df.shape[0] - sample_win ) / sample_stride) + 1)
    obs = 1
    for k in range(n_intervals):
        df.loc[k * sample_stride : k * sample_stride + sample_win,'observation'] = obs
        #df['observation'].iloc[k * dif : k * dif + sample_win] = obs
        obs += 1
    
    df = df.drop(df[df.observation == -1].index)
    
    return df

def input_features(df):
    data_x = []
    data_y = []
    for k in df.observation.unique():
        data_x.append(df.loc[df.observation == k].values[:, :-2].T)
        
        #data_y.append(df.loc[df.observation == k].values[:, -2:])

        obs = df.loc[df.observation == k].values[0, -1]
        label = 1 if(df.loc[df.observation == k].values[:, -2].sum() > 0) else 0
        
        data_y.append([obs, label])
                
        
    # concatenate along the rows axis

    data_x = np.stack(data_x, axis=0)
    data_x = data_x.astype(np.float32)
    data_y = np.stack(data_y, axis=0)
    
    return data_x, data_y 