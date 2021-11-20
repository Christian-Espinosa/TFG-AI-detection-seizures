import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from numpy.matlib import repmat

from collections import OrderedDict


# Fully Connected Block
def FC_block(n_inputs_loc, hidden_loc, n_output_loc,
                 relu_config=None,batch_config=None,p_drop_loc=0.1): 
    
    # Dictionary defining Block Architecture
    BlockArchitecture=[]
    hidden_loc.insert(0,n_inputs_loc)
    hidden_loc.append(n_output_loc)
    if relu_config==None:
        relu_config=repmat('no_relu',len(hidden_loc),1)
    if batch_config==None:
        batch_config=repmat('no_batch',len(hidden_loc),1)
    #Block Layers List
    for i in np.arange(len(hidden_loc)-1):
        BlockArchitecture.append(('linear'+str(i+1),
                                  nn.Linear(hidden_loc[i], hidden_loc[i+1])))
        BlockArchitecture.append(('drop'+str(i+1),nn.Dropout(p_drop_loc)))
        if(batch_config[i]=='batch'):
            BlockArchitecture.append(('batch'+str(i+1),nn.BatchNorm1d( hidden_loc[i+1])))
            
        
        if(relu_config[i]=='relu'):
            BlockArchitecture.append(('relu'+str(i+1),nn.ReLU(inplace=True)))
        
    linear_block_loc = nn.Sequential(
        OrderedDict(BlockArchitecture)
        )
    return linear_block_loc

# Convolutional Block 1D: Input should be (NSamp,1,NChanels,SignalLength)
# Convolves in the signal domain for each channel. All channel have equal kernels
# Uses conv2d with kernel=(1,sze) instead of conv1d to avoid transposing input data 
# If you want to use conv1d, data should be of shape (NSamp,SignalLength,NChanels)
#
# To convert from (NSamp,NChanels,SignalLength) to (NSamp,1,NChanels,SignalLength)
# use torch.unsqueeze(input_data, dim=1) 
    
def Conv1D_StandardBlock(n_inputs_loc, Nneurons_loc,kernel_size_loc, 
                 batch_config=None,p_drop_loc=0.0): 
    
    # Dictionary defining Block Architecture
    BlockArchitecture=[]
    Nneurons_loc.insert(0,n_inputs_loc)
#    Nneurons_loc.append(n_output_loc)

    if batch_config==None:
        batch_config=repmat('no_batch',len(Nneurons_loc),1)
    #Block Layers List
    for i in np.arange(len(Nneurons_loc)-1):
        BlockArchitecture.append(('conv'+str(i+1),
                                  nn.Conv2d(Nneurons_loc[i], Nneurons_loc[i+1],
                                            kernel_size=(1,kernel_size_loc[i]), 
                                            padding=(0,int(kernel_size_loc[i]-1)/2))))
        print(i)
        BlockArchitecture.append(('drop'+str(i+1),nn.Dropout(p_drop_loc)))
        
        if(batch_config[i]=='batch'):
           BlockArchitecture.append(('batch'+str(i+1),nn.BatchNorm2d( Nneurons_loc[i+1])))

        BlockArchitecture.append(('relu'+str(i+1),nn.ReLU(inplace=True)))
#    
#    BlockArchitecture.append( ('maxpool'+str(i+1),
#                              nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))))
            
    conv_block_loc = nn.Sequential(
        OrderedDict(BlockArchitecture)
        )
    return conv_block_loc
    


# =============================================================================
# Projection Blocks
# =============================================================================
    
# Projection of Input Channels
# projection input is (NSamp,NChanels (n_inputs_loc),SignalLength)
# projection output is (NSamp,NNeurons (n_output_loc),SignalLength)
#
def InputChannelProj(n_inputs_loc, n_output_loc=1,kernel_size=1,proj_type='AvgW'):
    
    
    if proj_type=='AvgW':
        projector = nn.Sequential(
            nn.Conv1d(n_inputs_loc, n_output_loc, kernel_size=1),
            nn.ReLU(inplace=True),
            )
    elif proj_type=='Avg':
        projector = nn.Sequential(
            nn.AvgPool2d((n_inputs_loc,1))
             )   
    elif proj_type=='ConvAvg':
        projector = nn.Sequential(
            nn.Conv1d(n_inputs_loc, n_output_loc, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            )
    return projector


# Projection of Output Channels
# projection input is (NSamp,NNeurons,NChanels,SignalLength), 
# with n_inputs_loc=(NChanels,SignalLength)
# projection output is  (NSamp,NNeurons (n_output_loc))


def OutPutNetProj_x(x,proj_type='AvgSCh'):
    
    n_output_loc=x.size(1)
    n_inputs_loc=(x.size(2),x.size(3))
    # Signal as Average and then channels as weighted average
    if proj_type=='AvgSAvgWCh':
        x=F.avg_pool2d(x,(1,n_inputs_loc[1])) # (NSamp,NNeurons,NChanels,1) 
        x= x.flatten(start_dim=2, end_dim=-1) # (NSamp,NNeurons,NChanels)
        x = torch.unsqueeze(x, dim=1) # (NSamp,1,NNeurons,NChanels)
        
        projector = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1,n_inputs_loc[0])),
            nn.ReLU(inplace=True),
            )
        x=projector(x) # (NSamp,1,NNeurons,1)

    # Averages signal and channels 
    elif proj_type=='AvgSCh':
        x=F.avg_pool2d(x,n_inputs_loc)
#        projector = nn.Sequential(
#            nn.AvgPool2d(n_inputs_loc)
#             )  
    # Weighted Average of signal and channels
    elif proj_type=='AvgWSCh':
        projector = nn.Sequential(
            nn.Conv2d(n_output_loc, n_output_loc, kernel_size=n_inputs_loc),
            nn.ReLU(inplace=True),
            )
        x=projector(x) # (NSamp,NNeurons,1,1)
        
    x=x.view(x.size(0),-1) # (NSamp,NNeurons)
    return x

def OutPutNetProj(n_inputs_loc,n_output_loc,proj_type='AvgSCh'):
    
#    n_output_loc=x.size(1)
#    n_inputs_loc=(x.size(2),x.size(3))
    # Signal as Average and then channels as weighted average
    if proj_type=='AvgSAvgWCh':
       
        projector = nn.Sequential(
            nn.AvgPool2d((1,n_inputs_loc[1])), # (NSamp,NNeurons,NChanels,1) 
            nn.Conv2d(n_output_loc, n_output_loc, kernel_size=(n_inputs_loc[0],1)),  # (NSamp,NNeurons,1,1)
            nn.ReLU(inplace=True),
            )
   
    # Averages signal and channels 
    elif proj_type=='AvgSCh':

        projector = nn.Sequential(
            nn.AvgPool2d(n_inputs_loc)
             )  
    # Weighted Average of signal and channels
    elif proj_type=='AvgWSCh':
        projector = nn.Sequential(
            nn.Conv2d(n_output_loc, n_output_loc, kernel_size=n_inputs_loc),
            nn.ReLU(inplace=True),
            )

    return projector