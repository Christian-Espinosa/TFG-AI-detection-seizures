#%%
from numpy.lib.type_check import imag
import pyedflib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os, sys
import lib.Format_edf_to_paquet as fra
import matplotlib.pyplot as plt

from scipy.signal import freqz
from scipy.signal import butter, lfilter
import timeit

import stdio 

#%%
print("Start")
start = timeit.default_timer()

name_edf = "chb01_03"
name_edf_seizures = "chb01_03"
#file_name = os.path.abspath(os.getcwd() + "/Data/edf/" + name + ".edf")
file_name = os.path.abspath("d:\\UAB\\4to\\TFG-AI-detection-seizures\\" + "Data/edf/" + name_edf + ".edf")
path_parquet = os.path.abspath("d:\\UAB\\4to\\TFG-AI-detection-seizures\\" + "Data/parquet/" + name_edf + ".parquet")
edf_f = pyedflib.EdfReader(file_name)
df, dic = fra.Create_parquet_from_edf(edf_f, path_parquet)

stop = timeit.default_timer()
print('Time: ', stop - start)  

#%%
print("Overall Plot")
start = timeit.default_timer()

plt.figure()
plt.plot(dic["FP1-F7"])
plt.title("Raw Data")
plt.draw()

stop = timeit.default_timer()
print('Time: ', stop - start)  

#%%

start = timeit.default_timer()

fs = 256
lowcut = 0.5
highcut = 50
print('{} to {} Plot'.format(lowcut, highcut))

y = fra.butter_bandpass_filter(dic["FP1-F7"], lowcut, highcut, fs, order=6)
plt.figure()
plt.plot(y, label='Filtered signal')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')
plt.title('{} to {} Plot'.format(lowcut, highcut))
plt.draw()

stop = timeit.default_timer()
print('Time: ', stop - start)  

#%%
f_range = 'theta'
print("{} Plot".format(f_range))
start = timeit.default_timer()


if f_range == 'theta':
    t_lowcut = 3.5
    t_highcut = 7.5
elif f_range == 'alpha':
    t_lowcut = 8
    t_highcut = 13
elif f_range == 'betal':
    t_lowcut = 8
    t_highcut = 13
elif f_range == 'betah':
    t_lowcut = 8
    t_highcut = 13

y = fra.butter_bandpass_filter(y, t_lowcut, t_highcut, fs, order=6)
x = fra.butter_bandpass_filter(dic["FP1-F7"], t_lowcut, t_highcut, fs, order=6)
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
#Seizures file
name_edf_seizures = "chb01_03s"
file_name = os.path.abspath("D:\\UAB\\4to\\TFG-AI-detection-seizures\\" + "Data/edf/" + name_edf_seizures + ".edf")
edf_f = pyedflib.EdfReader(file_name)
df = fra.set_seizure_labeling(df, edf_f)

plt.show()

# %%


# %%
