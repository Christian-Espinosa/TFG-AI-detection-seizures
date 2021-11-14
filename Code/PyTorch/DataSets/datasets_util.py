import os
import numpy as np
import pandas as pd

from sklearn import preprocessing

def scalers_fit(x_train):
    # source https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
    scalers = {}
    for i in range(x_train.shape[1]):
        scalers[i] = preprocessing.StandardScaler()
        # scalers[i] = preprocessing.MaxAbsScaler()
        # scalers[i] = preprocessing.MinMaxScaler()
        x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :])
    return x_train, scalers

def scalers_transform(scalers, x_test):
    for i in range(x_test.shape[1]):
        x_test[:, i, :] = scalers[i].transform(x_test[:, i, :])
    return x_test

def read_input_features(path, filename):
    filename_x = filename + '_data_x.npy'
    filename_y = filename + '_data_y.npy'
    data_x = np.load(os.path.join(path,filename_x), allow_pickle=True)
    data_y = np.load(os.path.join(path,filename_y), allow_pickle=True)
    return data_x, data_y


