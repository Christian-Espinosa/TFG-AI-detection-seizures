from os import path
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

def setLabels(dic, f, hz=256):
    #sets labels to a file
    dic['labels'] = np.zeros(len(dic[dic.keys()[-1]], dtype=np.int4))
    name = f[-12:] #Get last part of string
    name = name[:-4] #Delete .txt
    n = 1
    with open(f, 'r') as file:
        lines = file.readlines()
    for i in range(len(lines)):
        l = lines[i].strip()
        if l == "File Name: " + name + "_" + "{:02.0f}".format(n) + ".edf":
            i += 1
            if lines[i].strip() == "Number of Seizures in File: 0":
                return dic
            elif lines[i].strip()[:-1] == "Number of Seizures in File: ":
                for _ in range(int(lines[i].strip()[-1])):
                    i += 1
                    ini = int(l[i].strip()[:-8].replace('Seizure Start Time: ',''))*hz
                    i += 1
                    fi = int(l[i].strip()[:-8].replace('Seizure End Time: ',''))*hz
                    for x in range(ini, fi):
                        dic['labels'][x] = 1
                return dic
    return None

def saveToParquet(dic, path):
    pd.DataFrame.from_dict(dic).to_parquet(path)
