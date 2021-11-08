#%%
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


print("Start")
start = timeit.default_timer()

name = "chb01_01"
file_name = os.path.abspath(os.getcwd() + "/Data/edf/" + name + ".edf")
path_parquet = os.path.abspath(os.getcwd() + "/Data/parquet/" + name + ".parquet")
edf_f = pyedflib.EdfReader(file_name)
df, dic = fra.Create_parquet_from_edf(edf_f, path_parquet)
print(dic)
stop = timeit.default_timer()
print('Time: ', stop - start)  
print("Overall Plot")
start = timeit.default_timer()

plt.figure(1)
plt.plot(dic["FP1-F7"])
plt.draw()

stop = timeit.default_timer()
print('Time: ', stop - start)  
print("0.5 to 50 Plot")
start = timeit.default_timer()

fs = 256
lowcut = 0.5
highcut = 50

y = fra.butter_bandpass_filter(dic["FP1-F7"], lowcut, highcut, fs, order=6)
plt.plot(y, label='Filtered signal')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')
plt.draw()

stop = timeit.default_timer()
print('Time: ', stop - start)  
print("Theta Plot")
start = timeit.default_timer()

plt.figure(2)
t_lowcut = 4
t_highcut = 7

y = fra.butter_bandpass_filter(y, t_lowcut, t_highcut, fs, order=6)
x = fra.butter_bandpass_filter(dic["FP1-F7"], t_lowcut, t_highcut, fs, order=6)
plt.plot(x, label='From original signal')
plt.plot(y, label='From previous filter')
plt.xlabel('time (seconds)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')
plt.draw()


stop = timeit.default_timer()
print('Time: ', stop - start)  
plt.show()