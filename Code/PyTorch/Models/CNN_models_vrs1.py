import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

# OWN FUNCTIONS
"""
TFG: Check import has no problem (sintax)
"""
from eeg_models_init import *
from eeg_util_models import * 
from eeg_models_layers import *
# ========================================================================
# CONVOLUTIONAL BLOCK  
"""
TFG: Quan s'hagi testejat els models amb arquitectura fixada, aquesta funció no serà necessaria
"""
def ConVNet(n_features,ker_temp):
   #Definimos padding para que la imagen siga teniendo el mismo tamaño
   pad_temp=tuple(((np.array(ker_temp)-1)/2).astype(int))
   
   #
   block1 = nn.Sequential(
            #Configurar padding para no comerme los bordes
            nn.Conv2d(1, n_features, kernel_size=ker_temp, padding=pad_temp),
            #Regularizador, los que tengan probabilidad cerca de 0, no me salgan nan
            # Introduce 0 para regularizar - - > investiga
            nn.Dropout(0.1),
        #    nn.BatchNorm2d(n_features),
            nn.ReLU(inplace=True),
            
            # nn.Conv2d(n_features, n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(n_features, n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            # Reucir la dimensionalidad, extrae la informacion resumen de donde estan las formas buscadas en la imagen
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
            ).cuda()#REVISAR - Aplicar el tensor, compilació del bloc perque ho entengui el ordenador, de manera mas directa

   block2 = nn.Sequential(
            nn.Conv2d(n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.Dropout(0.1),
        #    nn.BatchNorm2d(2 * n_features),
            nn.ReLU(inplace=True),
            
            # nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
            ).cuda()

   block3 = nn.Sequential(
            nn.Conv2d(2 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.Dropout(0.1),
            #   nn.BatchNorm2d(4 * n_features),
            nn.ReLU(inplace=True),
            
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
            ).cuda()

   return (block1,block2,block3)


# 
# =============================================================================
#  spatiotemporal models with different option for combining channels data
# =============================================================================

# =============================================================================
# MODEL PROCESSING EACH CONCATENATION OF INPUT CHANNELS 
# =============================================================================

class CNN_ConcatInput(nn.Module):
    """

    Convolution across time of concatenated input channels
    1D convolutional network with concatenation of input channels.
    Convolutional blocks are given by net_params dictionary:
        
     1. convnet_params['kernel_size']=list of kernel sizes for each block
        convnet_params['Nneurons']=list of neurons for the layers of each block. One entry per block

        
        ex (2 blocks with 3 layers each).  
             convnet_params['kernel_size']=[ ]
             convnet_params['kernel_size'].append([3,3,3])
             convnet_params['kernel_size'].append([3,3,3])
             
             convnet_params['Nneurons']=[ ]
             convnet_params['Nneurons'].append([16,32,64])
             convnet_params['Nneurons'].append([128,128,128])

    2.  projlayer_params['proj_type']: type of projection
        projlayer_params['n_channels']: number of input channels
        projlayer_params['Nneurons']: number of neurons for first layer 
        if proj_type is 'Avg', omit parameter 
            
             projlayer_params=None
        
    3.  outputlayer_params['n_classes']=number of output classes

    """
    def __init__(self, projmodule_params,convnet_params,outputmodule_params):
        super().__init__()

        print('\tRunning class: ', self.__class__.__name__)
        ### PARAMETERS 
        
        # Net Parameters
        self.projmodule_params=projmodule_params
        self.convnet_params=convnet_params
        self.outputmodule_params=outputmodule_params
        
    
        
        ############# ARCHITECTURE
        ## input block: projection (None)
        
        ############# convolutional block for temporal signal processing
        """
            TFG: Cal posar aquest tros en aquest format i replicar-ho a tot arreu:       
        #        self.ConvBlockList=[]
        #        Nneurons_Block=convnet_params['Nneurons']
        #        Kernel_Block=convnet_params['kernel_size']
        #        
        #        for Nneurons_conv,kernel_size_conv in zip(Nneurons_Block,Kernel_Block):
        #            block=Conv1D_StandardBlock(NneuronsInput, Nneurons_conv.copy(),kernel_size_conv.copy(), 
        #                     batch_config=None,p_drop_loc=0.0)
        #            self.ConvBlockList.append(block)
         """
        kern_conv=convnet_params['kernel_size']  
        n_features=convnet_params['Nneurons']

        #Pytorch, define arquitectura basica de la red convolucional segun hyperparametros
        #Aruitectura para extraer las caracteristicas
        self.conv_net=ConVNet(n_features,kern_conv)

     
        ############ output block: projection channel (None), 
        ## projection signal (AvgPool), fc layers and classifier
        
        ### projector: has to be implemented in forward evaluation if
        ### you want independence from input signal size
        last_block=self.conv_net[-1]

        #Buscar que hace el sequencial!
        n_output_loc=last_block[0].out_channels

        #Que hacemos con las caracateristicas, como combinamos las caracts para clasificar
        self.outconv_proj= OutPutNetProj((1,0),n_output_loc)
        ### classification fc layer
        n_classes=outputmodule_params['n_classes']
        
        #Combinacion lineal de las caracteristicas
        self.fc_out = nn.Linear(n_output_loc, n_classes)

        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):

        # input is (NSamp, NChannels, LSignal) 
        #nSamples = finestres per subjecte
        #nChannels = 23 electrodes (como los metemos? juntos, por separado?)
        # LSignal=L (1 dim); LSignal=(W,H) (2 dim); LSignal=(W,H,L)

        ## Input Signal Projection: Concatenation
        x=x.view(x.size(0), -1) #[NSamples,NChannels*L]
       
        x = torch.unsqueeze(x, dim=1) # (N, 1, NChannels*L)
        #Trick to use conv2d operator instead of conv1
        x = torch.unsqueeze(x, dim=1) # (N, 1, 1,NChannels*L)  LSignal=(1,NChannels*L) 
        
        ## Convolutional Block: temporal signal features
         
        for block in self.conv_net:
            x=block(x)  # [N, NNeurons,1,L']
        

        ## Output Block
        ### projector: Channel (None), Signal (AvgPool)
        ### Dynamically adapt to any input length
        n_inputs_loc=(x.size(2),x.size(3))
        self.outconv_proj[0].kernel_size=n_inputs_loc
        self.outconv_proj[0].stride=n_inputs_loc
        x=self.outconv_proj(x)
        x = x.view(x.size(0), -1) # [N, NNeurons*1*1]
        
        # x=OutPutNetProj_x(x) # [N, NNeurons*1*1]
        ### classifier
        
        x = self.fc_out(x)
        return x

# =============================================================================
# MODELS PROCESSING EACH CHANNEL SEPARATELY AND WITH PROJECTION OF OUTPUT CHANNELS
# =============================================================================

# Former Seq_C2D_SpatTemp_v1 
# Output Projection: two convolutional layers. Elias design
class CNN_ProjOut_Conv(nn.Module):
    """

    Using 2D CNN - Convolution across time and then accross channels to 
    project network channel output
    Projection based on convolutions

    """
    def __init__(self, n_nodes=14, n_classes=2,n_features= 16):
        super().__init__()

        print('\tRunning class: ', self.__class__.__name__)
        
        #### PARAMETERS
        #  projection parameter: temporal signal output 
        self.signalout_kernel_dim = 4
        # convoluttion block parameters
        
        # classifier parameters
        
        ##### ARCHITECTURE
        ## input block: projection (None)
        
        ## convolutional block: temporal signal net
        self.temp_net=ConVNet(n_features,(1,3))
        
        ## Output Block

        ### projection of output signal channels
        last_block=self.temp_net[-1]
        self.spat_net = nn.Sequential(
            nn.Conv2d(last_block[0].out_channels, n_features * 4, kernel_size=(1,1), stride=(1,1)),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features * 4, n_features * 4, kernel_size=(n_nodes,1), stride=(1,1)),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            
            )

        ### classification fc layer
        self.fc_out = nn.Linear(n_features * 4 * self.signalout_kernel_dim, n_classes)

        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):

        # input is (N, NChannels, L)
        
        ## Input Signal Projection: None
        

        ## Convolutional Block: temporal signal features
        #Trick to use conv2d operator instead of conv1
        x = torch.unsqueeze(x, dim=1) # (N, 1, NChannels, L)
        x = self.temp_net(x) # [N, NNeurons, NChannels, L']
        
        ## Output Block
        ### projection of output channels
        x = self.spat_net(x) # [N, NNeurons', 1, L']
       
        ### classifier
        # to become invariant to input length:
        # average L' signal: # [N, NNeurons',signalout_kernel_dim]
        x = F.adaptive_avg_pool2d(x, (1, self.signalout_kernel_dim))
        # flattening
        x = x.view(x.size(0), -1) # [N, NNeurons'*signalout_kernel_dim]
        x = self.fc_out(x)
        return x


#######
# Former Seq_C2D
# Independent Processing of input channels with equal convolutional blocks
# Output Projection: average of concatenated out channels

class CNN_ProjOut_Concat(nn.Module):

    """
    Architecture : Implementation of the paper

        EEG-Based Spatio–Temporal Convolutional Neural Network for Driver Fatigue Evaluation
        Gao et al. 2019

    1D convolutional network with projection of channels output convolution.
    Each channel is processed independently using equal
    convolutional blocks 
    
    Convolutional blocks are given by net_params dictionary:
        convnet_params['kernel_size']=list of kernel sizes for each block
        convnet_params['Nneurons']=list of neurons for the layers of each block. One entry per block

        
        ex (2 blocks with 3 layers each).  
             convnet_params['kernel_size']=[ ]
             convnet_params['kernel_size'].append([3,3,3])
             convnet_params['kernel_size'].append([3,3,3])
             
             convnet_params['Nneurons']=[ ]
             convnet_params['Nneurons'].append([16,32,64])
             convnet_params['Nneurons'].append([128,128,128])

        projlayer_params['proj_type']: type of projection
        projlayer_params['n_channels']: number of input channels
        projlayer_params['Nneurons']: number of neurons for first layer 
        if proj_type is 'Avg', omit parameter
    
    
    
    """

    def __init__(self, n_nodes=14, n_classes=2, n_features= 16,n_features_out=50):
        super().__init__()

        print('running class ', self.__class__.__name__)

        
        # conv block parameters
        ker_temp = (1,3) # kernel for temporal dim
        #  temporal signal output projection parameter
            
        self.avg_pool_kernel_stride = 2
        
        ##### ARCHITECTURE
        ## input block: projection (None)
        
        ## Convolutional Block: temporal signal net
        self.temp_net=ConVNet(n_features,ker_temp)
        
        ## Output Block:
        ### projection of output channels
        """
        TFG: Cal posar això usant OutPutNetProj (if possible) i replicar a totes les classes que ho requereixin
        """
        last_block=self.temp_net[-1]
        self.outchannel_proj = nn.Sequential(
            # Elias avg pooling
            #nn.Linear(4 * n_features * n_nodes * self.avg_pool_kernel_stride, n_features_out),
            nn.Linear(last_block[0].out_channels * n_nodes, n_features_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)

            )
        
        ### classifier
        self.fc_out = nn.Linear(n_features_out, n_classes)

        # weight init
        init_weights_xavier_normal(self)
    

    def forward(self, x):

        x = torch.unsqueeze(x, dim=1) # (N, 1, NChannels, L)

        ## temporal features
        x = self.temp_net(x) # (N, NNeurons, NChannels, L')

        ## Output Block:
       
        
        ### signal pooling 
        ### to become invariant to input length
       
        n_inputs_loc=(x.size(2),x.size(3))
        x = F.avg_pool2d(x,(1,n_inputs_loc[1]))  # (N, NNeurons, NChannels, 1)
        
        # Elias avg pooling
#         temp_dim = x.shape[-1] # 10
#        x=F.avg_pool2d(x, kernel_size=(1, temp_dim // self.avg_pool_kernel_stride),
#                         stride=(1, temp_dim // self.avg_pool_kernel_stride))

        ### projection of output channels
        x = x.view(x.size(0), -1)   # (N, NNeurons*NChannels)
        x = self.outchannel_proj(x)  # (N, NNeuronsOut)
        
        ### classifier
        x = self.fc_out(x)

        return x

# Output Projection: weighted average of out channels
class CNN_ProjOut_AvgW(nn.Module):

    """
    Architecture : Implementation of the paper

        EEG-Based Spatio–Temporal Convolutional Neural Network for Driver Fatigue Evaluation
        Gao et al. 2019

    Input data is [N, features=14, timestep=40]
    """

    def __init__(self, n_nodes=14, n_classes=2, n_features= 16,n_features_out=50):
        super().__init__()

        print('running class ', self.__class__.__name__)
       # conv block parameters
        ker_temp = (1,3) # kernel for temporal dim
        #  temporal signal output projection parameter
        self.avg_pool_kernel_stride = 2
        
        ## temporal signal net
        self.temp_net=ConVNet(n_features,ker_temp)
        
        ## Output Block:
        """
        TFG: Cal posar això usant OutPutNetProj i replicar a totes les classes que ho requereixin
        """
    
        ## projection of output channels
        self.outchannel_proj = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1,n_nodes)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            )
        
        ## classifier
        last_block=self.temp_net[-1]
        self.fc = nn.Sequential(
            nn.Linear(last_block[0].out_channels, n_features_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)

            )
            
        self.fc_out = nn.Linear(n_features_out, n_classes)

      
        # weight init
        init_weights_xavier_normal(self)
    

    def forward(self, x):

        x = torch.unsqueeze(x, dim=1) # (N, 1, n_nodes, time_step)

        ## temporal features
        x = self.temp_net(x) # [NSamp,NNeurons, Nchanels,L']
  
    
        ## output channel projection
        # to become invariant to input length: AveragePool of L'
        x = F.avg_pool2d(x,(1,x.size(3))) # [NSamp,NNeurons, NChanels,1]
        x= x.flatten(start_dim=2, end_dim=-1)# [NSamp,NNeurons, NChanels]
        x = torch.unsqueeze(x, dim=1) # [NSamp,1,NNeurons, NChanels]
        
        x= self.outchannel_proj(x) # [NSamp,1,NNeurons,1]
        
        ## classifier
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc_out(x)

        return x

# =============================================================================
# MODELS WITH PROJECTION OF INPUT CHANNELS
# =============================================================================

# Former Seq_C2D_SpatTemp_v2
class CNN_ProjChannel(nn.Module):
    """

    Using 2D CNN - Projection of channels and then convolution of temporal projected signals
    Average pooling of CNN output to become invariant to input length

    """
    def __init__(self, n_nodes=14, n_classes=2,n_features= 16):
        super().__init__()

        print('\tRunning class: ', self.__class__.__name__)

        ##### ARCHITECTURE
        ## Projection Block
       
        """      
                # TFG: Cal posar el self.inchannel_proj en aquest format
                
        #        n_channels=projlayer_params['n_channels']
        #        NneuronsInput= projlayer_params['Nneurons']
        #        kernel_size=projlayer_params['kernel_size']
        #        proj_type=projlayer_params['proj_type']
        #        
        #        self.proj_layer= InputChannelProj(
        #                proj_type=proj_type,n_channels, 
        #                n_output_loc=NneuronsInput,kernel_size=kernel_size)
        """        
        self.inchannel_proj = nn.Sequential(
            nn.Conv2d(1, n_features, kernel_size=(n_nodes,1), stride=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            # nn.BatchNorm2d(n_features),
            )
       
        
         ## temporal signal net
        self.temp_net=ConVNet(n_features,(1,3))
        
        ## classifier
        #  self.fc_out = nn.Linear(n_features * 4 * 4, n_classes)
        last_block=self.temp_net[-1]
        self.fc_out = nn.Linear(last_block[0].out_channels , n_classes)
        
        
        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):

        # input is (N, NChannels, L)
        x = torch.unsqueeze(x, dim=1) # (N, 1, NChannels, L)

        ## Input Signal Projection: Weighted Average
        x = self.inchannel_proj(x) # [N, NProjNeurons, 1, L]

        ## temporal signal features
        x = self.temp_net(x) # [N, NNeurons, 1, L']
      
        ## classifier
        # to become invariant to input length
#        kernel_dim = 4
#        x = F.adaptive_avg_pool2d(x, (1, kernel_dim))
        n_inputs_loc=(x.size(2),x.size(3))
        x = F.avg_pool2d(x, (1, n_inputs_loc[1]))
        # flattening
        x = x.view(x.size(0), -1) # [N, NNeurons]
        x = self.fc_out(x)
        return x

# Per ML1 no sembla anar millor
class CNN_ProjChannel_v2(nn.Module):
    """

    Using 2D CNN - Projection of channels and then convolution of temporal projected signals
    Weighted pooling of CNN output to become invariant to input length
    """
    def __init__(self, n_nodes=14, n_classes=2,n_features= 16):
        super().__init__()

        print('\tRunning class: ', self.__class__.__name__)

       
        ##### ARCHITECTURE
        ## Projection Block
               
        """      
                # TFG: Cal posar el self.inchannel_proj en aquest format
                
        #        n_channels=projlayer_params['n_channels']
        #        NneuronsInput= projlayer_params['Nneurons']
        #        kernel_size=projlayer_params['kernel_size']
        #        proj_type=projlayer_params['proj_type']
        #        
        #        self.proj_layer= InputChannelProj(
        #                proj_type=proj_type,n_channels, 
        #                n_output_loc=NneuronsInput,kernel_size=kernel_size)
        """      
        self.inchannel_proj = nn.Sequential(
            nn.Conv2d(1, n_features, kernel_size=(n_nodes,1), stride=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            # nn.BatchNorm2d(n_features),
            )
        
         ## temporal signal net
         
                       
        """      
                # TFG: Cal posar el self.temp_net en aquest format
                
        #        self.ConvBlockList=[]
        #        Nneurons_Block=convnet_params['Nneurons']
        #        Kernel_Block=convnet_params['kernel_size']
        #        
        #        for Nneurons_conv,kernel_size_conv in zip(Nneurons_Block,Kernel_Block):
        #            block=Conv1D_StandardBlock(NneuronsInput, Nneurons_conv.copy(),kernel_size_conv.copy(), 
        #                     batch_config=None,p_drop_loc=0.0)
        #            self.ConvBlockList.append(block)
        """       
        self.temp_net=ConVNet(n_features,(1,3))
        

      #  self.fc_out = nn.Linear(n_features * 4 * 4, n_classes)

        self.fc_out = nn.Linear(n_features * 4 , n_classes)
        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):

        # input is (N, NChannels, L)
        x = torch.unsqueeze(x, dim=1) # (N, 1, NChannels, L)

        ## Input Signal Projection: Weighted Average
        x = self.inchannel_proj(x) # [N, NProjNeurons, 1, L]

        ## temporal signal features
        x = self.temp_net(x) # [N, NNeurons, 1, L']

        ## classifier
        # to become invariant to input length: 
        #        Weighted Projection
        n_inputs_loc=(x.size(2),x.size(3))
        projector = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(1,n_inputs_loc[1])),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
            ).cuda()
        
        x= x.flatten(start_dim=2, end_dim=-1) # [N, NNeurons,L']
        x = torch.unsqueeze(x, dim=1)  # [N, 1,NNeurons,L']
        x=projector(x) # [N, 1,NNeurons,1]
  
        # flattening
        x = x.view(x.size(0), -1) # [N,NNeurons]
        x = self.fc_out(x)
        return x

# =============================================================================
# CNN 2D
# =============================================================================


#################################################

class Seq_C1D(nn.Module):
    """
    Ensemble 1D - CNN

        Three layers:   16 - 32 - 64 layers
        Kernel size:    7, 5, 3, with stride=2, and padding acordingly.

        Revised :       05-08-21
    """
    def __init__(self, n_electrodes=14, n_classes=2):
        super().__init__()

        print('Running class ', self.__class__.__name__)

        n_features= 16 # change here
        self.features = nn.Sequential(

            nn.Conv1d(n_electrodes, n_features, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv1d(n_features, n_features * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv1d(n_features * 2, n_features * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            )

        self.fc_out = nn.Sequential(
            nn.Linear(n_features * 4, n_classes),
        )

        # weight init
        init_weights_xavier_normal(self)
    def get_embedding(self, x):
        x = self.features(x)
        kernel_size = x.shape[2]
        x = F.avg_pool1d(x, kernel_size) # [N, C, 1]
        x = x.view(x.size(0), -1) # [N, C]
        return x

    def forward(self, x):
        x = self.features(x) # [N, C, T], T = length/8

        # to become invariant to time length
        kernel_size = x.shape[2]
        x = F.avg_pool1d(x, kernel_size) # [N, C, 1]
        x = x.view(x.size(0), -1) # [N, C]

        x = self.fc_out(x)
        return x

# =============================================================================

#   ENSEMBLE MODELS: PROCESSING OF EACH CHANNEL USING DIFFERENT KERNELS 
#   PROJECTION OF ENSEMBLE OUTPUT        
# =============================================================================
class Seq_C1D_Ensemble(nn.Module):
    """
    Ensemble 1D - CNN

        Three layers:   16 - 32 - 64 layers
        Kernel size:    7, 5, 3, with stride=2, and padding acordingly.

        Revised :       05-08-21
    """
    def __init__(self, n_electrodes=14,  n_classes=2):
        super().__init__()

        in_features = 1
        n_features = 16

        
        self.id_electrodes = np.arange(n_electrodes)
 
        print('Running class ', self.__class__.__name__)
        print('id_electrodes \t----> ', self.id_electrodes)

        self.n_electrodes = len(self.id_electrodes)

        self.model_arr = nn.ModuleList(nn.ModuleList([]) for _ in range(self.n_electrodes))
        for i in range(self.n_electrodes):
            self.model_arr[i].append(self.layer_conv(in_features, n_features))
            self.model_arr[i].append(self.layer_linear(n_features, n_classes))

        self.fc_out = nn.Sequential(
            nn.Linear(n_features * 4, n_classes), # feature projections
            )

        self.projector = nn.Sequential(
            nn.Conv1d(self.n_electrodes, 1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            )

        # weight initizalization
        init_weights_xavier_normal(self)

    def layer_conv(self, n_inputs, n_outputs):
        conv_block = nn.Sequential(
            # architecture 1
            nn.Conv1d(n_inputs, n_outputs, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv1d(n_outputs , n_outputs * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv1d(n_outputs * 2, n_outputs * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            )
        return conv_block

    def layer_linear(self, n_inputs, n_classes):
        linear_block = nn.Sequential(
             nn.Linear(n_inputs * 4, n_classes),
            )
        return linear_block

    def forward(self, x):

        out = []
        for i in range(self.n_electrodes):
            # features
            data = x[:,i,:].unsqueeze(1) # [N, 1, 40]
            data = self.model_arr[i][0](data)
            kernel_size = data.shape[2] # to become invariant to time length
            data = F.avg_pool1d(data, kernel_size) # [N, 128, 1]
            data = data.view(data.size(0), -1) # [N, 128]
            out.append(data)

        out = torch.stack(out, dim=1) # [N, 14, 128] FEATURES
        # either
        # out = out.sum(1) # voting - Accuracy 0.65, 50-ep
        # or
        # out = out.view(out.size(0), -1) # concat Accuracy 0.59, 50-ep
        # or

        out = self.projector(out) # [N, 1, 128]  projector Accuracy 0.76, 50-ep, to 0.77 (+/- 0.016)
        out = out.view(out.size(0), -1) # [N, 128]
        out = self.fc_out(out)
        return out

