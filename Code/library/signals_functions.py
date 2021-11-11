# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 18:06:08 2021

@author: AURA
"""
import os
import pandas as pd
import numpy as np
from scipy import signal

from eeg_globals import *

def filt_signal(proj_dir,eeg_df,dic_filt_opts):
        
    labels = list(eeg_df.columns)
    
    # filter signals by IQR[dataset,setphase] and median filter
    #IQR filtering
    print(dic_filt_opts['IQRtype'])
    if dic_filt_opts['IQRtype'] == 'old':
        IQRs = pd.read_pickle(os.path.join(proj_dir,'input_features/IQR'))
    elif dic_filt_opts['IQRtype'] == 'new':
        IQRFile='input_features/IQR'+'_'+str(int(dic_filt_opts['q']*100))
        IQRs = pd.read_pickle(os.path.join(proj_dir,IQRFile))
        IQRs=IQRs[dic_filt_opts['IQRTh']]

    
    
    eeg_pow_filt  = []
    
    for phase in range(1, 3): # high results
        print('phase    ' + str(phase))
    
        sub = eeg_df.loc[ (eeg_df.phase == phase), labels].copy()
        sub = sub.reset_index(drop=True)
        # filtering
        meta = sub.iloc[:, 70:].values
        data = sub.iloc[:, :70].values

        if dic_filt_opts['per_phases']:
            th_up_all = IQRs[(dic_filt_opts['datafiltset'],phase)]
        else:
            th_up_all = IQRs[(dic_filt_opts['datafiltset'],dic_filt_opts['setphase'])]
        m_thresh = np.repeat([np.array(th_up_all)], data.shape[0], axis=0)
        mask = data > m_thresh
        data[mask] = m_thresh[mask]/2.
    
        #median filter applying
        for rr in range(data.shape[1]): # by colums (70 cols = 14 channesl * 5 waves)
            data[:, rr] = signal.medfilt(data[:, rr], kernel_size=3)
        
        df = pd.DataFrame(np.concatenate((data, meta), axis=1), columns=labels)
        eeg_pow_filt.append(df)
    
    eeg_pow_filt  = pd.concat(eeg_pow_filt, axis=0, ignore_index=True)
    
    return eeg_pow_filt

def cut_signal(eeg_df,dic_cut):
    labels = list(eeg_df.columns) # (275812, 102)


    sample_win = int(PSF * dic_cut['window'])
    sample_over = int(PSF * dic_cut['overlap'])
    sample_stride = sample_win - sample_over 
    
    # To data, add a column observation based on phase
    print('split data into observations')
    #eeg_power_window = pd.DataFrame()
    
    all_tests = eeg_df.test.unique() # three tests
    all_phases = eeg_df.phase.unique() # two phases
    
    eeg_power_window = []
    for subject in eeg_df.subject.unique():
        print(subject)
        for test in all_tests:
            print('     ' + str(test))
            for phase in all_phases:
                print('         ' + str(phase))
    
                df = eeg_df.loc[(eeg_df.subject==subject) &
                                (eeg_df.test==test) &
                                (eeg_df.phase==phase)].copy()
                df = df.reset_index(drop=True)
                df = df.drop([ 'subject', 'test', 'phase'], axis=1 )
    
                n_intervals = int(np.floor(( df.shape[0] - sample_win ) / sample_stride) + 1) 
                obs = 1
                for k in range(n_intervals):
                    data = df.iloc[k * sample_stride : k * sample_stride + sample_win].copy()
                    data = data.reset_index(drop=True)
                    data['subject'] = subject
                    data['test'] = test
                    data['phase'] = phase
                    data['observation'] = obs
                    eeg_power_window.append(data)
                    obs += 1
                    del data
                del df
    
    eeg_power_window  = pd.concat(eeg_power_window, axis=0, ignore_index=True)
    return eeg_power_window

def cut_signal_simulator(eeg_df,dic_cut):
    labels = list(eeg_df.columns) 


    sample_win = int(PSF * dic_cut['window'])
    sample_over = int(PSF * dic_cut['overlap'])
    dif = sample_win - sample_over
    
    # To data, add a column observation based on phase
    print('split data into observations')
    #eeg_power_window = pd.DataFrame()
    
    all_flights = eeg_df.flight.unique() # three tests
    all_tests = eeg_df.test.unique() # three tests
    all_phases = eeg_df.phase.unique() # two phases
    
    eeg_power_window = []
    for subject in eeg_df.subject.unique():
        print(subject)
        for flight in all_flights:
            print('     ' + str(flight))
            for test in all_tests:
                print('     ' + str(test))
                for phase in all_phases:
                    print('         ' + str(phase))
        
                    df = eeg_df.loc[(eeg_df.subject==subject) &
                                    (eeg_df.flight==flight) &
                                    (eeg_df.test==test) &
                                    (eeg_df.phase==phase)].copy()
                    df = df.reset_index(drop=True)
                    df = df.drop([ 'subject', 'test', 'phase', 'perceived_difficulty', 'theoretical_difficulty'], axis=1 )
        
                    n_intervals = df.shape[0] // dif - 1
                    obs = 1
                    for k in range(n_intervals):
                        data = df.iloc[k * dif : k * dif + sample_win].copy()
                        data = data.reset_index(drop=True)
                        data['subject'] = subject
                        data['test'] = test
                        data['phase'] = phase                        
                        data['observation'] = obs
                        data['flight'] = flight
                        eeg_power_window.append(data)
                        obs += 1
                        del data                    
                    del df
        
    eeg_power_window  = pd.concat(eeg_power_window, axis=0, ignore_index=True)
    return eeg_power_window

def input_features(eeg_df):
    labels = list(eeg_df.columns)

    # define the ndexes of cols of interest
    pow_theta = labels[0 : 70 : 5]
    pow_alpha = labels[1 : 70 : 5]
    pow_betal = labels[2 : 70 : 5]
    pow_betah = labels[3 : 70 : 5]
    pow_gamma = labels[4 : 70 : 5]
    
    dic_pow_waves = {'theta' : pow_theta,
                 'alpha' : pow_alpha,
                 'betal' : pow_betal,
                 'betah' : pow_betah,
                 'gamma' : pow_gamma,
                 }
    all_pow_waves = list(dic_pow_waves.keys())
    all_tests = eeg_df.test.unique()
    all_phases = eeg_df.phase.unique()
    
    # data and labels are into separated files
    data_x = []
    data_y = []
    for subject in eeg_df['subject'].unique():
            print(subject)
            for test in all_tests:
                print('     ' + str(test))
                for phase in all_phases:
                    print('         ' + str(phase))
    
                    signal_arr = []
                    target_arr = ''
                    for idx, wave in enumerate(all_pow_waves):
    
                        data_labels = dic_pow_waves[wave]
                        meta_labels = labels[(14 * 5): ] # elias, this change according the number of channels
                        labels_sel = data_labels + meta_labels
    
                        df = eeg_df.loc[(eeg_df.subject==subject) &
                                             (eeg_df.test==test) &
                                             (eeg_df.phase==phase), labels_sel].copy()
                        df = df.reset_index(drop=True)
    
                        signal = []
                        target = []
                        for k in df.observation.unique():
                            # from [timesteps, n_features] to [n_features, timesteps]
                            x = df.loc[df.observation == k].values[:, :len(labels_sel) - 4].T
                            y = df.loc[df.observation == k].values[0, len(labels_sel) - 4 : len(labels_sel) - 4 + 4] # GET only test and phase
                            signal.append(x)
                            target.append(y)
                            del x, y
                        del df
    
                        signal_arr.append(np.array(signal))
                        if idx == 0:
                            target_arr = np.array(target) # get labels from the first passing
                        del signal, target
    
                    # stack signals [signals, n_features, timesteps]
                    signal_arr = np.stack((signal_arr), axis=1)
                    #target_arr = np.array(target_arr)
                    if signal_arr.ndim == 4:
                        data_x.append(signal_arr)
                        data_y.append(target_arr)
                    del signal_arr, target_arr
    
    # concatenate along the rows axis
    data_x = np.concatenate(data_x, axis=0) # (n_samples, n_waves, n_nodes, time_step)
    data_x = data_x.astype(np.float32)
    data_y = np.concatenate(data_y, axis=0) # (n_samples, 4)
    return data_x, data_y

def input_features_simulator(eeg_df):
    labels = list(eeg_df.columns)

    # define the ndexes of cols of interest
    pow_theta = labels[0 : 70 : 5]
    pow_alpha = labels[1 : 70 : 5]
    pow_betal = labels[2 : 70 : 5]
    pow_betah = labels[3 : 70 : 5]
    pow_gamma = labels[4 : 70 : 5]
    
    dic_pow_waves = {'theta' : pow_theta,
                 'alpha' : pow_alpha,
                 'betal' : pow_betal,
                 'betah' : pow_betah,
                 'gamma' : pow_gamma,
                 }
    all_pow_waves = list(dic_pow_waves.keys())
    all_tests = eeg_df.test.unique()
    all_phases = eeg_df.phase.unique()
    
    
    # data and labels are into separated files
    data_x = []
    data_y = []
    for subject in eeg_df['subject'].unique():
            print(subject)
            for flight in eeg_df.loc[(eeg_df.subject==subject)].flight.unique():
                print('     ' + str(flight))
                for test in all_tests:
                    print('     ' + str(test))
                    for phase in all_phases:
                        print('         ' + str(phase))
        
                        signal_arr = []
                        target_arr = ''
                        for idx, wave in enumerate(all_pow_waves):
        
                            data_labels = dic_pow_waves[wave]
                            meta_labels = labels[(14 * 5): ] # elias, this change according the number of channels
                            labels_sel = data_labels + meta_labels
        
                            df = eeg_df.loc[(eeg_df.subject==subject) &
                                                 (eeg_df.test==test) &
                                                 (eeg_df.flight==flight) &
                                                 (eeg_df.phase==phase), labels_sel].copy()
                            df = df.reset_index(drop=True)
                            df.subject.replace({subject: subject+flight},inplace=True)
        
                            signal = []
                            target = []
                            for k in df.observation.unique():
                                # from [timesteps, n_features] to [n_features, timesteps]
                                x = df.loc[df.observation == k].values[:, :len(labels_sel) - 4].T
                                y = df.loc[df.observation == k].values[0, len(labels_sel) - 4 : len(labels_sel) - 4 + 4] # GET only test and phase
                                signal.append(x)
                                target.append(y)
                                del x, y
                            del df
        
                            signal_arr.append(np.array(signal))
                            if idx == 0:
                                target_arr = np.array(target) # get labels from the first passing
                            del signal, target
        
                        # stack signals [signals, n_features, timesteps]
                        signal_arr = np.stack((signal_arr), axis=1)
                        #target_arr = np.array(target_arr)
                        if signal_arr.ndim == 4:
                            data_x.append(signal_arr)
                            data_y.append(target_arr)
                        del signal_arr, target_arr
    
    # concatenate along the rows axis
    data_x = np.concatenate(data_x, axis=0) # (n_samples, n_waves, n_nodes, time_step)
    data_x = data_x.astype(np.float32)
    data_y = np.concatenate(data_y, axis=0) # (n_samples, 4)
    return data_x, data_y


def input_features_elias(eeg_df):
    """
    Organize the eeg_df dataframe onto a numpy structure ready to feed into
    the model

    Returns
    -------
    data_x : numpy
        data of signals [n_samples, n_waves, n_nodes, timesteps]

    data_y : numpy
        metadata of signals [n_samples, subject...]

    """
    labels = eeg_df.columns.to_list()
    total_pow_nodes = len([i for i in labels if i.startswith('POW')])
    if total_pow_nodes != 70:
        print('The number of nodes does not match the Emotiv 14 headset')
        return

    # data and labels are into separated files
    data_x = []
    data_y = []
    for subject in eeg_df.subject.unique():
            print(subject)
            for test in eeg_df.test.unique():
                print('     ' + str(test))
                for phase in eeg_df.phase.unique():
                    print('         ' + str(phase))

                    signal_arr = []
                    target_arr = ''
                    for idx, wave in enumerate(all_pow_waves):

                        data_labels = dic_pow_waves[wave]
                        meta_labels = labels[total_pow_nodes : ] # [70 : ]
                        labels_sel = data_labels + meta_labels

                        df = eeg_df.loc[(eeg_df.subject==subject) &
                                             (eeg_df.test==test) &
                                             (eeg_df.phase==phase), labels_sel].copy()
                        df = df.reset_index(drop=True)

                        signal = []
                        target = []

                        for k in df.observation.unique():

                            x = df.loc[df.observation == k].values[:, :len(labels_sel) - len(meta_labels)].T  # data
                            y = df.loc[df.observation == k].values[0 , len(labels_sel) - len(meta_labels) : ] # metadata

                            signal.append(x)
                            target.append(y)
                            del x, y
                        del df

                        signal_arr.append(np.array(signal))
                        if idx == 0:
                            target_arr = np.array(target) # save labels from only the first passing
                        del signal, target

                    # stack signals [signals, wave, n_nodes, timesteps]
                    signal_arr = np.stack((signal_arr), axis=1)
                    if signal_arr.ndim == 4:
                        data_x.append(signal_arr)
                        data_y.append(target_arr)
                    del signal_arr, target_arr

    # concatenate along the rows axis
    data_x = np.concatenate(data_x, axis=0) # (n_samples, n_waves, n_nodes, time_step)
    data_x = data_x.astype(np.float32)
    data_y = np.concatenate(data_y, axis=0) # (n_samples, ...)
    return data_x, data_y 