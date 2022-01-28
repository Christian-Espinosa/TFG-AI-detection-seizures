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



#%%

#IMPORTANT JERARCHY NEEDED TO EXECUTE THE CODE
#repo
#/DataSetTFG
# --> /CHB-MIT
# --> /CVC

#Define Variables and Read edf
single_execution = False

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
dic_cut = {'window' : 10, # window size seconds
                'overlap' : 0} # overlapping seconds

#Errors
err=[]

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
    path_numpy = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + "/numpy/" + name_edf)    
    file_summary = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + "/" + summary + ".txt")

    #%%
    print("Start of subject: {} file: {}".format(subj, name_edf))
    start = timeit.default_timer()

    edf_f = pyedflib.EdfReader(file_name)
    _ , dic = fra.Create_parquet_from_edf(edf_f, path_parquet)

    
    stop = timeit.default_timer()
    print('Time get edf: ', stop - start)  
    #%%
    #Define max BandWidth and Theta
    #dic = setBandwidth(dic, dic_band_definitions['maxrange'], hz)
    dic = fra.setBandwidth(dic, dic_band_definitions[f_range], hz)
    

    # %%
    start = timeit.default_timer()
    if(fra.check_23electrodes(file_summary, summary, channels)):
        print("Subject electrodes wihout interruptions")
    
    dic = fra.setLabels(dic, file_summary, file)
    
    if dic == None:
        print("Error in labeling!")

    if show_plots:
        fra.Rawplot1Channel(dic)
        fra.plotea(dic)

    fra.saveToParquet(dic, path_parquet)
    fra.chunkData(path_parquet, dic_cut)
    fra.saveToNumpy(path_parquet, path_numpy)
    
    
        
    stop = timeit.default_timer()
    print('Time label and save: ', stop - start)  
    plt.show()
    # %%
else:


    # %%
    n_subjects = 10
    show_plots = False
    print("Start {} subjects".format(n_subjects))
    start = timeit.default_timer()

    #For every subject
    for i in range(1,n_subjects+1):

        subj = "chb{:02.0f}".format(i)
        summary = subj + "-summary"
        elem = os.listdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + "/edf"))

        #Only considers files .edf in the directory of the edfs
        for j in range(len(elem)):
            print('################')
            if elem[j][-4:] ==  ".edf":
                name_edf = elem[j][:-4]
                f = int(name_edf[-1])
                #name_edf = elem[j][:-4]
                file_name = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + "/edf/" + name_edf + ".edf")
                path_parquet = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/chb01/parquet/" + name_edf + ".parquet")
                path_numpy = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + "/numpy/" + name_edf)
                file_summary = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + '/' + summary + ".txt")
                
                edf_f = pyedflib.EdfReader(file_name)
                _ , dic = fra.Create_parquet_from_edf(edf_f, path_parquet)

                #Filter data
                dic = fra.setBandwidth(dic, dic_band_definitions[f_range], hz)

                #Data is as expected
                if(fra.check_23electrodes(file_summary, summary, channels)):
                    print("{} is ok".format(name_edf, f))
                
                #Set labels to GT
                dic = fra.setLabels(dic, file_summary, f)
                if dic == None:
                    print("Error in labeling!")

                #Save all data
                fra.saveToParquet(dic, path_parquet)
                fra.chunkData(path_parquet, dic_cut)
                try:
                    fra.saveToNumpy(path_parquet, path_numpy)
                except:
                    print('ERROR: File not saved as numpy: ', name_edf)
                    err.append(name_edf)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    print('ERRORS IN EXECUTION:')
    for i in err:
        print(i)