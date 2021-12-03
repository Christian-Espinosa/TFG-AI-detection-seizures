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
    start = timeit.default_timer()

    plt.figure()
    plt.plot(dic[elec])
    plt.title("Raw Data")
    plt.draw()

    stop = timeit.default_timer()
    print('Time: ', stop - start)  

def setBandwidth(dic, range, hz, elec = "FP1-F7"):

    start = timeit.default_timer()
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

    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    return dic

def check_23electrodes(f, name, c):
    with open(f, 'r') as file:
        i=1
        find = False
        for l in file:
            l = l.rstrip()
            
            if l == "Channel {}: {}".format(i, c[i-1]):
                find = True
            else:
                if find and i>=1 and i<24:
                    print("Electrodes are not ok in {}".format(name))
            if find:
                i = i+1


#%%
#Define Variables and Read edf
print("Start")
start = timeit.default_timer()

name_edf = "chb01_03"
summary = "chb01-summary"

#Paths
#file_name = os.path.abspath(os.getcwd() + "/Data/edf/" + name + ".edf")
file_name = os.path.abspath("D:\UAB\4to\DataSetTFG" + "\CHB-MIT\edf" + name_edf + ".edf")
path_parquet = os.path.abspath("D:\UAB\4to\DataSetTFG" + "\CHB-MIT\parquet" + name_edf + ".parquet")
file_summary = os.path.abspath("D:\UAB\4to\DataSetTFG" + "\CHB-MIT"+ summary + ".txt")

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



edf_f = pyedflib.EdfReader(file_name)
_ , dic = fra.Create_parquet_from_edf(edf_f, path_parquet)

stop = timeit.default_timer()
print('Time: ', stop - start)  

#%%
Rawplot1Channel(dic)

#%%
#Define max BandWidth and Theta
dic = setBandwidth(dic, dic_band_definitions['maxrange'], hz)
dic = setBandwidth(dic, dic_band_definitions[f_range], hz)

# %%
check_23electrodes(file_summary, summary, channels)
dic = fra.setLabels(dic, file_summary, name_edf)
fra.saveToParquet(dic)
plt.show()
