#Utility Tools
import numpy as np
import pandas as pd
import mne
import os, sys

#View Tools
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Data input
dataset_path = os.getcwd() + "/Data/chb01_01.edf"
data = mne.io.read_raw_edf(dataset_path)
wave_set = data.get_data()
#print(f'Dataset dimensions (rows / columns): {wave_set.shape}')
#print(f'Dataset attributes: {wave_set.keys()}')
# showing rows 1 to 5 from the dataset
#wave_set.head()


# you can get the metadata included in the file and a list of all channels:
print(data.info)
print(data.ch_names)
#join
hyper_element = " ".join(data.ch_names)

print(hyper_element)