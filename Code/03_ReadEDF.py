#%%
from numpy.lib.type_check import imag
import pyedflib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os, sys
import library.Format_edf_to_parquet as fra
import matplotlib.pyplot as plt

from scipy.signal import freqz
from scipy.signal import butter, lfilter
import timeit

import stdio 
#%%

def Rawplot1Channel(dic, elec = "FP1-F7"):
    #Plot 1 electrode no filtering
    print("Overall Plot")
    plt.figure()
    plt.plot(dic[elec])
    plt.title("Raw Data")
    plt.draw()

def setBandwidth(dic, range, hz, elec = "FP1-F7"):
    #sets bandwidth of all electrodes, prints one
    
    print('{} to {} Plot'.format(range[0], range[1]))
    for k, v in dic.items():
        dic[k] = fra.butter_bandpass_filter(v ,range[0], range[1], hz, order=6)
    plt.figure()
    plt.plot(dic[elec], label='Filtered signal')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.title('{} to {} Plot Electrode: {}'.format(range[0], range[1], elec))
    plt.draw()

    return dic

def check_23electrodes(f, name, c):
    with open(f, 'r') as file:
        i=1
        find = False
        for l in file:
            l = l.strip()
            if i < 24:
                if l == "Channel {}: {}".format(i, c[i-1]):
                    find = True
                else:
                    if find and i>=1 and i<24:
                        print("Electrodes are not ok in {}".format(name))
                        return False
                if find:
                    i = i+1
    return True


#%%

#IMPORTANT JERARCHY NEEDED TO EXECUTE THE CODE
#repo
#/DataSetTFG
# --> /CHB-MIT
# --> /CVC

#Define Variables and Read edf
single_execution = True

#FILTERING FEATURES
f_range = 'theta'
hz = 256
dic_band_definitions = { 'delta' : [0.5, 4],
                        'theta' : [4, 8],
                        'alpha' : [8, 12],
                        'beta' : [12, 30],
                        'gamma' : [30, 45],
                        'maxrange' : [0.5, 50]}
#23 channels
channels = ["FP1-F7","F7-T7","T7-P7","P7-O1","FP1-F3","F3-C3","C3-P3","P3-O1","FP2-F4","F4-C4","C4-P4","P4-O2","FP2-F8","F8-T8","T8-P8","P8-O2","FZ-CZ","CZ-PZ","P7-T7","T7-FT9","FT9-FT10","FT10-T8","T8-P8"]


if single_execution:
    #%%
    n_subj = 1
    file = 3
    show_plots = True

    name_edf = "chb{:02.0f}_{:02.0f}".format(n_subj, file)
    summary = "chb{:02.0f}-summary".format(n_subj)
    subj = "chb{:02.0f}".format(n_subj)
    #Paths
    file_name = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + "/edf/" + name_edf + ".edf")
    path_parquet = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + "/parquet/" + name_edf + ".parquet")
    file_summary = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + "/" + summary + ".txt")

    #%%
    print("Start of subject: {} file: {}".format(subj, name_edf))
    start = timeit.default_timer()

    edf_f = pyedflib.EdfReader(file_name)
    _ , dic = fra.Create_parquet_from_edf(edf_f, path_parquet)

    Rawplot1Channel(dic)
    stop = timeit.default_timer()
    print('Time get edf: ', stop - start)  
    #%%
    #Define max BandWidth and Theta
    dic = setBandwidth(dic, dic_band_definitions['maxrange'], hz)
    dic = setBandwidth(dic, dic_band_definitions[f_range], hz)

    # %%
    start = timeit.default_timer()
    if(check_23electrodes(file_summary, summary, channels)):
        print("Subject electrodes wihout interruptions")
    
    dic = fra.setLabels(dic, file_summary, file)
    if dic == None:
        print("Error in labeling!")

    fra.saveToParquet(dic, path_parquet)
    if show_plots:
        plt.show()
    stop = timeit.default_timer()
    print('Time label and save: ', stop - start)  

    # %%
else:
    # %%
    n_subjects = 1
    print("Start {} subjects".format(n_subjects))
    start = timeit.default_timer()

    #For every subject
    for i in range(n_subjects):
        subj = "chb{:02.0f}".format(i)
        summary = subj + "-summary"
        elem = os.listdir(os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + "/edf"))
        #Only considers files .edf in the directory of the edfs
        for j in range(len(elem)):
            if elem[j][-4:] ==  ".edf":
                name_edf = subj + "_chb{:02.0f}".format(i) + "_{:02.0f}".format(j)
                #name_edf = elem[j][:-4]
                file_name = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + "/edf/" + name_edf + ".edf")
                path_parquet = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/chb01/parquet/" + name_edf + ".parquet")
                file_summary = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + summary + ".txt")
                
                edf_f = pyedflib.EdfReader(file_name)
                _ , dic = fra.Create_parquet_from_edf(edf_f, path_parquet)
                Rawplot1Channel(dic)
                dic = setBandwidth(dic, dic_band_definitions['maxrange'], hz)
                dic = setBandwidth(dic, dic_band_definitions[f_range], hz)
                if(check_23electrodes(file_summary, summary, channels)):
                    print("Subject: {} is ok".format(i))
                dic = fra.setLabels(dic, file_summary)
                if dic == None:
                    print("Error in labeling!")

                fra.saveToParquet(dic, path_parquet)
                plt.show()

    stop = timeit.default_timer()
    print('Time: ', stop - start)