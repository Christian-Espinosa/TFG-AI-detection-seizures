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
#Define Variables and Read edf
print("Start")
start = timeit.default_timer()

name_edf = "chb01_03"
name_edf_seizures = "chb01_03"
#file_name = os.path.abspath(os.getcwd() + "/Data/edf/" + name + ".edf")
file_name = os.path.abspath("D:\UAB\4to\DataSetTFG" + "\CHB-MIT\edf" + name_edf + ".edf")
path_parquet = os.path.abspath("D:\UAB\4to\DataSetTFG" + "\CHB-MIT\parquet" + name_edf + ".parquet")
dic_band_definitions = { 'delta' : [0.5, 4],
                        'theta' : [4, 8],
                        'alpha' : [8, 12],
                        'beta' : [12, 30],
                        'gamma' : [30, 45],
                        'maxrange' : [0.5, 50]}

#FILTERING FEATURES

f_range = 'theta'
hz = 256


edf_f = pyedflib.EdfReader(file_name)
df, dic = fra.Create_parquet_from_edf(edf_f, path_parquet)

stop = timeit.default_timer()
print('Time: ', stop - start)  

#%%
#Plot 1 electrode no filtering
print("Overall Plot")
start = timeit.default_timer()

plt.figure()
plt.plot(dic["FP1-F7"])
plt.title("Raw Data")
plt.draw()

stop = timeit.default_timer()
print('Time: ', stop - start)  

#%%
#Define max BandWidth
start = timeit.default_timer()


print('{} to {} Plot'.format(dic_band_definitions['maxrange'][0], dic_band_definitions['maxrange'][1]))

y = fra.butter_bandpass_filter(dic["FP1-F7"], dic_band_definitions['maxrange'][0], dic_band_definitions['maxrange'][1], hz, order=6)
plt.figure()
plt.plot(y, label='Filtered signal')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')
plt.title('{} to {} Plot'.format(dic_band_definitions['maxrange'][0], dic_band_definitions['maxrange'][1]))
plt.draw()

stop = timeit.default_timer()
print('Time: ', stop - start)  

#%%


print("{} Plot".format(f_range))
start = timeit.default_timer()

y = fra.butter_bandpass_filter(y, dic_band_definitions[f_range][0], dic_band_definitions[f_range][1], hz, order=6)
x = fra.butter_bandpass_filter(dic["FP1-F7"], dic_band_definitions[f_range][0], dic_band_definitions[f_range][1], hz, order=6)
plt.figure()
plt.plot(x, label='From original signal')
plt.plot(y, label='From previous filter')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')
plt.title("{} Plot".format(f_range))
plt.draw()


stop = timeit.default_timer()
print('Time: ', stop - start)  

# %%

#LABELING

#Seizures file
file_name = os.path.abspath("D:\\UAB\\4to\\TFG-AI-detection-seizures\\" + "Data/edf/" + name_edf_seizures + ".edf.seizures")
x = open(file_name, "rb")

ofset = int(str(int(x.read(39), 2)) + str(int(x.read(42), 2)), 2)

#Puede ayudar
#https://www.mathworks.com/matlabcentral/answers/225716-how-i-can-read-chb01_03-edf-seizures-file-from-chb-mit-database-in-matlab-as-i-am-using-this-file-f

#df = fra.set_seizure_labeling(df, edf_f)

plt.show()

# %%


# %%
