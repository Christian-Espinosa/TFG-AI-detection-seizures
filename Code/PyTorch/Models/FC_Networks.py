import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
import numpy as np
from numpy.matlib import repmat

from Networks_Initialization import *

from collections import OrderedDict

def linear_block(n_inputs_loc, hidden_loc, n_output_loc,
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
        if(relu_config[i]=='relu'):
            BlockArchitecture.append(('relu'+str(i+1),nn.ReLU(inplace=True)))
        if(batch_config[i]=='batch'):
            BlockArchitecture.append(('batch'+str(i+1),nn.BatchNorm1d( hidden_loc[i+1])))
            
    linear_block_loc = nn.Sequential(
        OrderedDict(BlockArchitecture)
        )
    return linear_block_loc

class My_ResNetBlock(nn.Module):

    def __init__(self,input_channels, hidden=[128,128],output_channels=256,p_drop=0.1,use_1x1conv=True):
        super().__init__()
        
        
        model= linear_block(input_channels, 
                            hidden, output_channels,p_drop_loc=p_drop)
        
        # aux=list(model.named_modules())
        # self.linear_block=nn.Sequential(
        #     OrderedDict(aux[1:-2]+list(aux[-1])))
        # Alternativa
        aux=dict(model.named_modules())
        aux.pop('') # Remove model
        aux.pop('relu'+str(len(hidden)+1)) #Remove last relu layer
        
        self.linear_block=nn.Sequential(OrderedDict(aux))
        
        if use_1x1conv:
            self.conv_1x1 = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.conv_1x1 = None

    def forward(self, x):

        out = self.linear_block(x)
        
        if self.conv_1x1:
            x = self.conv_1x1(x)
        out += x
        return F.relu(out)

# =============================================================================
#
# =============================================================================

##      FULLY CONNECTED MODELS
# 1. MultiLayer Perceptron
#          
class Seq_NN(nn.Module):
    """
    MultiLayer Perceptron: 
    Netwotk with n_hidden layers with architecture linear+drop+relu+batch
     Constructor Parameters:
           n_inputs: dimensionality of input features (n_channels * n_features , by default) 
                     n_channels (=14), number of sensors or images for each case
                     n_features(=40), number of features for each n_channels
           n_classes: number of output classes (=3, by default)
           hidden(=[128,128], default): list with the number of neurons for each hidden layer
           p_drop(=0.1, default): probability for Drop layer (=0, no drop is performed)

    """
    def __init__(self, n_inputs=14*40, n_classes=3,hidden=[128],
                 relu_config=None,batch_config=None,p_drop=0.1):
        super().__init__()

        print('running class ', self.__class__.__name__)
        self.n_inputs=n_inputs
        self.hidden=hidden.copy()
        self.n_classes=n_classes
        self.p_drop=p_drop
        
        self.linear_block= linear_block(n_inputs, self.hidden.copy(), n_classes, 
                                                 relu_config=relu_config, 
                                                 batch_config=batch_config,
                                                 p_drop_loc=p_drop)
        self.fc_out=nn.Sigmoid()
        # weight init
        init_weights_xavier_normal(self)

    def get_embedding(self, x):
        return self.linear_block(x)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        x = self.fc_out(x)
        return x

# 2. MultiLayer Perceptron Ensemble combining Features
#  
class Seq_NN_Ensemble_By_Features(nn.Module):
    """
    MultiLayer Perceptron Ensemble that combines the output 
    features of n_channel MLP networks with architecture Seq_NN
    Constructor Parameters:
           n_channels (=14, default): number of sensors or images for each case
           n_features(=40, default): number of features for each n_channels
           n_classes(=3, by default): number of output classes 
           hidden(=[128,128], default): list with the number of neurons for each hidden layer
           p_drop(=0.1, default): probability for Drop layer (=0, no drop is performed)

    """

    def __init__(self, n_channels=14, n_features=40, n_classes=3, batch_config=None,
                 hidden=[128],p_drop=0.1,relu_config=None):
        super().__init__()

        
        self.hidden=hidden.copy()
        self.n_nets = n_channels
        self.n_features=n_features
        self.n_classes=n_classes
        self.sensors = np.arange(n_channels)

         # Generate n_nets classification networks (self.model_arr), 
         # one for each input sensor
     
        n_output=self.hidden.pop()
   
        self.model_arr = nn.ModuleList([ ])
        for i in range(self.n_nets):
            self.model_arr.append(linear_block(
                n_features, self.hidden.copy(), n_output,
                relu_config=relu_config, 
                batch_config=batch_config,p_drop_loc=p_drop))
        self.hidden.append(n_output)    
        
        # Agregate self.model_arr output to be the input of the 
        # network that fuses all sensors
        self.fc_out = nn.Sequential(
          
            nn.Linear(self.n_nets * n_output, n_classes),
            nn.Sigmoid(),
            )
        
        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):
       
        out = []
        for i in range(self.n_nets):
            data = x[:,i,:]
            data = self.model_arr[i](data) # the best at 200 epochs
            out.append(data)

        out = torch.stack(out, dim=1) # [N, models, outs]
        out = out.view(out.size(0), -1) # [N, models * outs]
        out = self.fc_out(out)
        return out

# 3. MultiLayer Perceptron Ensemble combining Networks Probabilities
# 
class Seq_NN_Ensemble_By_Probabilities(nn.Module):
    """
    MultiLayer Perceptron Ensemble that combines the output 
    probabilities of n_channel MLP networks

    Constructor Parameters:
           n_channels (=14, default): number of sensors or images for each case
           n_features(=40, default): number of features for each n_channels
           n_classes(=3, by default): number of output classes 
           hidden(=[128,128], default): list with the number of neurons for each hidden layer
           p_drop(=0.1, default): probability for Drop layer (=0, no drop is performed)


    """

    def __init__(self, n_channels=14, n_features=40, n_classes=3, 
                 hidden=[128],p_drop=0.1,relu_config=None,batch_config=None):
        super().__init__()

        self.hidden=hidden.copy()
        self.n_nets = n_channels
        self.n_features=n_features
        self.n_classes=n_classes
        self.sensors = np.arange(n_channels)

        
      #  print('running class ', self.__class__.__name__) # agregado
      #  print('Input Sensors----> ', self.sensors) # agregado
      
        # Generate n_nets classification networks (self.model_arr), one for each input sensor
        self.model_arr = nn.ModuleList([ ])
        for i in range(self.n_nets):
            self.model_arr.append(linear_block(n_features, 
                                               self.hidden.copy(), n_classes,
                                                relu_config=relu_config, 
                                                batch_config=batch_config,
                                                p_drop_loc=p_drop))
        # Agregate self.model_arr output to be the input of the 
        # network that fuses all sensors
        self.fc_out = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(self.n_nets * n_classes, n_classes),
            nn.Sigmoid(),
            )

        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):
      
        out = []
        for i in range(self.n_nets):
            data = x[:,i,:]
            data = self.model_arr[i](data) # the best at 200 epochs
            out.append(data)

        out = torch.stack(out, dim=1) # [N, models, outs]
        out = out.view(out.size(0), -1) # [N, models * outs]
        out = self.fc_out(out)
        return out

# =============================================================================

class MY_ResNet(nn.Module):
    """
    Input data is [N, features=14, timestep=40]

    Architecture : Residual
        https://arxiv.org/pdf/1611.06455.pdf

    """
    def __init__(self, n_features=14, n_classes=3):
        super().__init__()

        print('running class ', self.__class__.__name__)

        n_filters = 128 # the best at 128, block of 2
        self.block_1 = My_ResNetBlock(n_features, n_filters)
        self.block_2 = My_ResNetBlock(n_filters, n_filters * 2)
        self.block_3 = My_ResNetBlock(n_filters * 2, n_filters * 2)

        self.fc_out = nn.Sequential(
            nn.Linear(n_filters * 2, n_classes),
        )

        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        k_size = x.shape[2]
        x = F.avg_pool1d(x, k_size)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x












