# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 13:22:56 2021

@author: debora
"""

import numpy as np
from random import shuffle

def ConstantDataSet(n_classes,L,NSamp):
    
    data_x=np.empty((0,L))
    data_y=np.empty((0,1))
    
    basic_x=np.ones((NSamp,L))
    basic_y=np.ones((NSamp,1))
      
    for kclass in np.arange(n_classes):
        data_x=np.append(data_x,basic_x*kclass)
        data_y=np.append(data_y,basic_y*kclass)
    
    shuffle(data_x)
    shuffle(data_y)
    
    return data_x,data_y