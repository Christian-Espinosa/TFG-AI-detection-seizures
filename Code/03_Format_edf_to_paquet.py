import pyedflib
import numpy as np
import os, sys
import warnings
warnings.filterwarnings("ignore")

file_name = os.getcwd() + "/Data/chb01_01.edf"
f = pyedflib.EdfReader(file_name)
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))

for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)

print(signal_labels)
print(sigbufs)