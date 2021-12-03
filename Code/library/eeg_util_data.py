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

def remove_subject_data(data_x, data_y, blacklist = ['alejandro', 'sanket']):
    for subject in blacklist:
        chosen_rows = data_y[:, 0] == subject
        data_x = data_x[~chosen_rows]
        data_y = data_y[~chosen_rows]
    return data_x, data_y

def select_subject_train_test_data(data_x, data_y, subject):

    if type(subject) == list:
        chosen_rows = np.zeros((data_y.shape[0],), dtype=bool)
        for sub in subject:
            chosen_rows |= data_y[:, 0] == sub
    else:
        chosen_rows = data_y[:, 0] == subject

    #chosen_rows = data_y[:, 0] == subject # code improved to select more than one subject at each time
    train_data_x = data_x[~chosen_rows, :]
    train_data_y = data_y[~chosen_rows, :]
    test_data_x = data_x[chosen_rows, :]
    test_data_y = data_y[chosen_rows, :]
    return train_data_x, train_data_y, test_data_x, test_data_y

def select_and_encode_data_for_pretrained(data_x_train, data_y_train, EXP_TYPE):

    # create a new label to ensure to select right labels
    # if data_y_train.shape[1]==5:
    #     data_y_train = pd.DataFrame(data_y_train, columns=('subject', 'test', 'phase', 'observation', 'status'))
    # elif data_y_train.shape[1]==4:
    #     data_y_train = pd.DataFrame(data_y_train, columns=('subject', 'test', 'phase', 'observation'))
    # else:
    #     print('The label dimension does not match!!!')

    data_y_train = pd.DataFrame(data_y_train, columns=('subject', 'test', 'phase', 'observation'))
    data_y_train['label'] = data_y_train.apply(lambda row: int(str(row.test) + str(row.phase)), axis=1)

    # select only the labes to be used in the training/test
    if EXP_TYPE == 'BLs_WL2':
        data_y_train = data_y_train.loc[
                        (data_y_train.label==11) |
                        (data_y_train.label==21) |
                        (data_y_train.label==31) |
                        
                        (data_y_train.label==22)  , :]
    
    elif EXP_TYPE == 'BLs_WLs':
        data_y_train = data_y_train.loc[
                        (data_y_train.label==11) |
                        (data_y_train.label==21) |
                        (data_y_train.label==31) |                        
                        
                        (data_y_train.label==12) |
                        (data_y_train.label==22) |
                        (data_y_train.label==32)  ,  :]
    
    elif EXP_TYPE == 'BLs_WL2union3':
        data_y_train = data_y_train.loc[
                        (data_y_train.label==11) |
                        (data_y_train.label==21) |
                        (data_y_train.label==31) |                        
                        
                        (data_y_train.label==22) |                        
                        (data_y_train.label==32) , :]
    
    elif EXP_TYPE == 'WL1_WL2':
        data_y_train = data_y_train.loc[
                        (data_y_train.label==12) |
                        (data_y_train.label==22) , :]
    
    elif EXP_TYPE == 'WL1_WL2_WL3':
        data_y_train = data_y_train.loc[
                        (data_y_train.label==12) |
                        (data_y_train.label==22) |
                        (data_y_train.label==32) , :]
    
    elif EXP_TYPE == 'BLs_WL2_WL3':
        data_y_train = data_y_train.loc[
                        (data_y_train.label==11) |
                        (data_y_train.label==21) |
                        (data_y_train.label==31) |                        
                        
                        (data_y_train.label==22) |                        
                        (data_y_train.label==32) , :]
    
    elif EXP_TYPE == 'BLs_WL1_WL2_WL3':
        data_y_train = data_y_train.loc[
                        (data_y_train.label==11) |
                        (data_y_train.label==21) |
                        (data_y_train.label==31) |                        
                        
                        (data_y_train.label==12) |
                        (data_y_train.label==22) |
                        (data_y_train.label==32)  ,  :]
    else:
        print('Not implemented experiment type!')

    # select the data from the data_x
    chosen_rows = list(data_y_train.index)
    data_x_train = np.take(data_x_train, chosen_rows, axis=0)
    data_y_train = data_y_train.reset_index(drop=True)

    # encode labels
    encoder_dic = dict_label_for_testing(EXP_TYPE)
    data_y_train['label'] = data_y_train.apply(lambda row: encoder_dic[row.label], axis=1)
    data_y_train = data_y_train.values

    return data_x_train, data_y_train

        
def dict_label_for_testing(EXP_TYPE):
    """
    Elias : Encode labels to train
    """
    if EXP_TYPE == 'BLs_WL2':
        encoder_dic = {
                11 : 0,
                22 : 1,
                21 : 0,
                31 : 0             
                }
        
    elif EXP_TYPE == 'BLs_WLs':
        encoder_dic = {
                11 : 0,                
                21 : 0,
                31 : 0,
                12 : 1,                
                22 : 1,
                32 : 1,
                }
    
    elif EXP_TYPE == 'BLs_WL2union3':
        encoder_dic = {
                11 : 0,                
                21 : 0,
                31 : 0,        
                22 : 1,
                32 : 1,
                } 
    
    elif EXP_TYPE == 'WL1_WL2':
        encoder_dic = {
                12 : 0,                
                22 : 1,                
                }        
        
    elif EXP_TYPE == 'WL1_WL2_WL3':
        encoder_dic = {
                12 : 0,                
                22 : 1,                
                32 : 2
                }        
    
    elif EXP_TYPE == 'BLs_WL2_WL3':
        encoder_dic = {
                11 : 0,                
                21 : 0,
                31 : 0,                         
                22 : 1,
                32 : 2,
                }
    elif EXP_TYPE == 'BLs_WL1_WL2_WL3':
        encoder_dic = {
                11 : 0,                
                21 : 0,
                31 : 0,
                12 : 1,                
                22 : 2,
                32 : 3,
                }

    else:
        print("error in encoder_dic definition")

    return encoder_dic


# =============================================================================
# format manually the class code
# =============================================================================

def select_data_for_experiment(data_x, data_y):

    # create a new label to ensure to select right labels
    # if data_y.shape[1]==5:
    #     data_y = pd.DataFrame(data_y, columns=('subject', 'test', 'phase', 'observation', 'status'))
    # elif data_y.shape[1]==4:
    #     data_y = pd.DataFrame(data_y, columns=('subject', 'test', 'phase', 'observation'))
    # else:
    #     print('The label dimension does not match!!!')

    data_y = pd.DataFrame(data_y, columns=('subject', 'test', 'phase', 'observation'))
    data_y['label'] = data_y.apply(lambda row: int(str(row.test) + str(row.phase)), axis=1)

    # select the test and phases of interest and extract data based on their code indexes
    # which data will be used in the train and test

    data_y = data_y.loc[(data_y.label==11) |
                            (data_y.label==12) |
                            (data_y.label==21) |
                            (data_y.label==22) |
                            (data_y.label==31) |
                            (data_y.label==32), ]

    # select the data from the data_x
    chosen_rows = list(data_y.index)
    data_x = np.take(data_x, chosen_rows, axis=0)
    data_y = data_y.reset_index(drop=True)
    data_y = data_y.values

    return data_x, data_y

def label_train_data_for_model(train_data_x, train_data_y, EXP_TYPE):

    x_train = train_data_x
    y_train = train_data_y[:, -1].astype(np.int64)

    # encoding manually
    if EXP_TYPE == 'WL1_WL2_WL3':

        y_train[ y_train == 12] = 0
        y_train[ y_train == 22] = 1
        y_train[ y_train == 32] = 2

        # select only the label of interest
        chosen_rows = (y_train == 0) | (y_train == 1) | (y_train == 2)
        x_train = x_train[chosen_rows]
        y_train = y_train[chosen_rows]

    elif EXP_TYPE == 'BLs_WL1_WL2_WL3':

        y_train[ y_train == 11] = 0
        y_train[ y_train == 21] = 0
        y_train[ y_train == 31] = 0

        y_train[ y_train == 12] = 1
        y_train[ y_train == 22] = 2
        y_train[ y_train == 32] = 3

        # select only the label of interest
        chosen_rows = (y_train == 0) | (y_train == 1) | (y_train == 2) | (y_train == 3)
        x_train = x_train[chosen_rows]
        y_train = y_train[chosen_rows]

    elif EXP_TYPE == 'BLs_WL2_WL3':

        y_train[ y_train == 11] = 0
        y_train[ y_train == 21] = 0
        y_train[ y_train == 31] = 0

        y_train[ y_train == 22] = 1
        y_train[ y_train == 32] = 2

        # select only the label of interest
        chosen_rows = (y_train == 0) | (y_train == 1) | (y_train == 2)
        x_train = x_train[chosen_rows]
        y_train = y_train[chosen_rows]

    elif EXP_TYPE == 'WL1_WL2':

        y_train[ y_train == 12] = 0
        y_train[ y_train == 22] = 1

        # select only the label of interest
        chosen_rows = (y_train == 0) | (y_train == 1)
        x_train = x_train[chosen_rows]
        y_train = y_train[chosen_rows]

    elif EXP_TYPE == 'BLs_WL2union3':
        y_train[ y_train == 11] = 0
        y_train[ y_train == 21] = 0
        y_train[ y_train == 31] = 0

        y_train[ y_train == 22] = 1
        y_train[ y_train == 32] = 1

        # select only the label of interest
        chosen_rows = (y_train == 0) | (y_train == 1)
        x_train = x_train[chosen_rows]
        y_train = y_train[chosen_rows]

    elif EXP_TYPE == 'BLs_WLs':
        y_train[ y_train == 11] = 0
        y_train[ y_train == 21] = 0
        y_train[ y_train == 31] = 0

        y_train[ y_train == 12] = 1
        y_train[ y_train == 22] = 1
        y_train[ y_train == 32] = 1

        # select only the label of interest
        chosen_rows = (y_train == 0) | (y_train == 1)
        x_train = x_train[chosen_rows]
        y_train = y_train[chosen_rows]

    elif EXP_TYPE == 'BLs_WL2':

        y_train[ y_train == 11] = 0
        y_train[ y_train == 21] = 0
        y_train[ y_train == 31] = 0
        y_train[ y_train == 22] = 1

        # select only the label of interest
        chosen_rows = (y_train == 0) | (y_train == 1)
        x_train = x_train[chosen_rows]
        y_train = y_train[chosen_rows]

    elif EXP_TYPE == 'BL1_WL2':

        y_train[ y_train == 11] = 0
        y_train[ y_train == 22] = 1

        # select only the label of interest
        chosen_rows = (y_train == 0) | (y_train == 1)
        x_train = x_train[chosen_rows]
        y_train = y_train[chosen_rows]

    elif EXP_TYPE == 'BL2_WL2':

        y_train[ y_train == 21] = 0
        y_train[ y_train == 22] = 1

        # select only the label of interest
        chosen_rows = (y_train == 0) | (y_train == 1)
        x_train = x_train[chosen_rows]
        y_train = y_train[chosen_rows]

    elif EXP_TYPE == 'BL3_WL2':

        y_train[ y_train == 31] = 0
        y_train[ y_train == 22] = 1

        # select only the label of interest
        chosen_rows = (y_train == 0) | (y_train == 1)
        x_train = x_train[chosen_rows]
        y_train = y_train[chosen_rows]

    else:
        print('labeling type not found for the experiment type!')

    return x_train, y_train

def label_test_data_for_model(test_data_x, test_data_y, EXP_TYPE):

    x_test = test_data_x
    y_test = test_data_y[:, -1].astype(np.int64)

     # encoding manually
    if EXP_TYPE == 'WL1_WL2_WL3':

        y_test[ y_test == 12] = 0
        y_test[ y_test == 22] = 1
        y_test[ y_test == 32] = 2

        chosen_rows = (y_test == 0) | (y_test == 1) | (y_test == 2)

        x_other = x_test[~chosen_rows]
        y_other = y_test[~chosen_rows]

        x_test = x_test[chosen_rows]
        y_test = y_test[chosen_rows]

    elif EXP_TYPE == 'BLs_WL1_WL2_WL3':

        y_test[ y_test == 12] = 1
        y_test[ y_test == 22] = 2
        y_test[ y_test == 32] = 3

        chosen_rows = (y_test == 1) | (y_test == 2) | (y_test == 3)
        x_other = x_test[~chosen_rows]
        y_other = y_test[~chosen_rows]

        y_test[ y_test == 11] = 0
        y_test[ y_test == 21] = 0
        y_test[ y_test == 31] = 0

    elif EXP_TYPE == 'BLs_WL2_WL3':

        y_test[ y_test == 22] = 1
        y_test[ y_test == 32] = 2

        chosen_rows = (y_test == 1) | (y_test == 2)
        x_other = x_test[~chosen_rows]
        y_other = y_test[~chosen_rows]

        y_test[ y_test == 11] = 0
        y_test[ y_test == 21] = 0
        y_test[ y_test == 31] = 0

        # discard no used WL1
        chosen_rows = (y_test == 0) | (y_test == 1) | (y_test == 2)
        x_test = x_test[chosen_rows]
        y_test = y_test[chosen_rows]

    elif EXP_TYPE == 'WL1_WL2':

        y_test[ y_test == 12] = 0
        y_test[ y_test == 22] = 1

        chosen_rows = (y_test == 0) | (y_test == 1)
        x_other = x_test[~chosen_rows]
        y_other = y_test[~chosen_rows]

        # discard no used BLs and WL3
        chosen_rows = (y_test == 0) | (y_test == 1)
        x_test = x_test[chosen_rows]
        y_test = y_test[chosen_rows]


    elif EXP_TYPE == 'BLs_WL2union3':

        # we need to evaluate the individual performance
        x_other = x_test.copy()
        y_other = y_test.copy()

        y_test[ y_test == 11] = 0
        y_test[ y_test == 21] = 0
        y_test[ y_test == 31] = 0

        y_test[ y_test == 22] = 1
        y_test[ y_test == 32] = 1

        # discard no used WL1
        chosen_rows = (y_test == 0) | (y_test == 1)
        x_test = x_test[chosen_rows]
        y_test = y_test[chosen_rows]

    elif EXP_TYPE == 'BLs_WLs':

        # we need to evaluate the individual performance
        x_other = x_test.copy()
        y_other = y_test.copy()

        y_test[ y_test == 11] = 0
        y_test[ y_test == 21] = 0
        y_test[ y_test == 31] = 0

        y_test[ y_test == 12] = 1
        y_test[ y_test == 22] = 1
        y_test[ y_test == 32] = 1

    elif EXP_TYPE == 'BLs_WL2':
        # we need to evaluate the individual performance
        y_test[ y_test == 22] = 1

        chosen_rows =  (y_test == 1)
        x_other = x_test[~chosen_rows].copy()
        y_other = y_test[~chosen_rows].copy()

        y_test[ y_test == 11] = 0
        y_test[ y_test == 21] = 0
        y_test[ y_test == 31] = 0

        # select only the label of interest
        chosen_rows = (y_test == 0) | (y_test == 1)
        x_test = x_test[chosen_rows]
        y_test = y_test[chosen_rows]

    elif EXP_TYPE == 'BL1_WL2':
        # we need to evaluate the individual performance
        y_test[ y_test == 11] = 0
        y_test[ y_test == 22] = 1

        chosen_rows = (y_test == 0) | (y_test == 1)
        x_other = x_test[~chosen_rows].copy()
        y_other = y_test[~chosen_rows].copy()

        # select only the label of interest
        x_test = x_test[chosen_rows]
        y_test = y_test[chosen_rows]

    elif EXP_TYPE == 'BL2_WL2':
        # we need to evaluate the individual performance
        y_test[ y_test == 21] = 0
        y_test[ y_test == 22] = 1

        chosen_rows = (y_test == 0) | (y_test == 1)
        x_other = x_test[~chosen_rows].copy()
        y_other = y_test[~chosen_rows].copy()

        # select only the label of interest
        x_test = x_test[chosen_rows]
        y_test = y_test[chosen_rows]

    elif EXP_TYPE == 'BL3_WL2':

        # we need to evaluate the individual performance
        y_test[ y_test == 31] = 0
        y_test[ y_test == 22] = 1

        chosen_rows = (y_test == 0) | (y_test == 1)
        x_other = x_test[~chosen_rows].copy()
        y_other = y_test[~chosen_rows].copy()

        # select only the label of interest
        x_test = x_test[chosen_rows]
        y_test = y_test[chosen_rows]

    else:
        print('Labeling type not found for the experiment type!')

    return x_test, y_test, x_other, y_other

def select_data_Aura(data_x, data_y,EXP_TYPE):

    if data_y.shape[1] != 4:
        print('the dataset does not match the required shape [subject, test, phase, observation]!')
        return

    data_y = pd.DataFrame(data_y, columns=('subject', 'test', 'phase', 'observation'))
    data_y['label'] = data_y.apply(lambda row: int(str(row.test) + str(row.phase)), axis=1)

    # ['BLs_WL2','WL1_WL2_WL3','BLs_WL2_WL3','BLs_WL1_WL2_WL3']
    # select only the labels to be used in the training/test
    if EXP_TYPE == 'BLs_WL2':
        data_y = data_y.loc[ (data_y.label==11) | (data_y.label==21) | (data_y.label==31) | (data_y.label==22), : ]
    elif EXP_TYPE == 'WL1_WL2_WL3':
        data_y = data_y.loc[ (data_y.label==12) | (data_y.label==22) | (data_y.label==32), : ]
    elif EXP_TYPE == 'BLs_WL2_WL3':
        data_y = data_y.loc[ (data_y.label==11) | (data_y.label==21) | (data_y.label==31) | (data_y.label==22) | (data_y.label==32), : ]
        
    # select the data from the data_x
    chosen_rows = list(data_y.index)
    data_x = np.take(data_x, chosen_rows, axis=0)
    data_y = data_y.reset_index(drop=True)

    encoder_dic = dict_label_for_testing(EXP_TYPE)
        
    data_y['label'] = data_y.apply(lambda row: encoder_dic[row.label], axis=1)
    data_y = data_y.values

    return data_x, data_y
# =============================================================================
# seriousgame
# =============================================================================

def seriousgame_select_data_for_experiment(data_x, data_y, EXP_TYPE):

    if data_y.shape[1] != 4:
        print('the dataset does not match the required shape [subject, test, phase, observation]!')
        return

    data_y = pd.DataFrame(data_y, columns=('subject', 'test', 'phase', 'observation'))
    data_y['label'] = data_y.apply(lambda row: int(str(row.test) + str(row.phase)), axis=1)

    # select only the labes to be used in the training/test
    if EXP_TYPE == 'WL1_WL2':
        data_y = data_y.loc[ (data_y.label==12) | (data_y.label==22), : ]
    else:
        print('Not implemented experiment type!')


    # select the data from the data_x
    chosen_rows = list(data_y.index)
    data_x = np.take(data_x, chosen_rows, axis=0)
    data_y = data_y.reset_index(drop=True)

    # encode labels
    encoder_dic = {
                12 : 0,
                22 : 1
                }
    data_y['label'] = data_y.apply(lambda row: encoder_dic[row.label], axis=1)
    data_y = data_y.values

    return data_x, data_y

def seriousgame_select_data_Deb(data_x, data_y,EXP_TYPE):

    if data_y.shape[1] != 4:
        print('the dataset does not match the required shape [subject, test, phase, observation]!')
        return

    data_y = pd.DataFrame(data_y, columns=('subject', 'test', 'phase', 'observation'))
    data_y['label'] = data_y.apply(lambda row: int(str(row.test) + str(row.phase)), axis=1)

    # select only the labes to be used in the training/test
    if EXP_TYPE == 'BLsin_WLsin':
        data_y = data_y.loc[ (data_y.label==11) | (data_y.label==12), : ]
    elif EXP_TYPE == 'BL_WL':
        data_y = data_y.loc[ (data_y.label==11) | (data_y.label==12) | (data_y.label==21) | (data_y.label==22), : ]
    elif EXP_TYPE == 'WL1_WL2':
        data_y = data_y.loc[ (data_y.label==12) | (data_y.label==22), : ]
        
    # select the data from the data_x
    chosen_rows = list(data_y.index)
    data_x = np.take(data_x, chosen_rows, axis=0)
    data_y = data_y.reset_index(drop=True)

    # encode labels

    if EXP_TYPE == 'BLsin_WLsin':
        encoder_dic = {
                    11 : 0,
                    12 : 1
                    }
    elif EXP_TYPE == 'BL_WL':
        print(EXP_TYPE)
        encoder_dic = {
            21 : 0,
            11 : 0,
            12 : 1,
            22 : 1
            }
    elif EXP_TYPE == 'WL1_WL2':
        encoder_dic = {
            12 : 0,
            22 : 1
            }
        
    data_y['label'] = data_y.apply(lambda row: encoder_dic[row.label], axis=1)
    data_y = data_y.values

    return data_x, data_y

# =============================================================================
# seriousgame
# =============================================================================
def FRAM_select_data(data_x,data_y):
    
    

    data_y = pd.DataFrame(data_y, columns=('subject', 
                                           'dummy','test','interruptions','event',
                                           'phase', 'observation'))
    data_y['label'] = data_y.apply(lambda row: int(str(row.test) + str(row.phase)), axis=1)

    
def wasim_ytrue(data_x, data_y):



    data_y = pd.DataFrame(data_y[:,0:5], columns=('subject', 'test', 'phase', 'observation','label'))
   
    data_y = data_y.loc[ (data_y.label>-1), : ]

    # select the data from the data_x
    chosen_rows = list(data_y.index)
    data_x = np.take(data_x, chosen_rows, axis=0)
    data_y = data_y.reset_index(drop=True)

    # encode labels

    encoder_dic = {
        0 : 0,
        1 : 0,
        2 : 1,
        3 : 1
        }
    data_y['label'] = data_y.apply(lambda row: encoder_dic[row.label], axis=1)
    data_y = data_y.values

    return data_x, data_y