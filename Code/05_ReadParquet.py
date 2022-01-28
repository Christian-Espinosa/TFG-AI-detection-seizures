import pandas as pd
import os
import numpy as np



subj = 'chb' + "{:02.0f}".format(1)
numpys = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + '/numpy/')
            
#Numpys
npys = os.listdir(numpys)
for file in range(0,int(len(npys)),2):

    print("Loading:".format(npys[file]))
    data_x = np.load(os.path.join(numpys,npys[file]), allow_pickle=True)
    data_y = np.load(os.path.join(numpys,npys[file+1]), allow_pickle=True)
    print('file {}: {}'.format(npys[file], data_x.shape))
    print('file {}: {}'.format(npys[file+1], data_y.shape))