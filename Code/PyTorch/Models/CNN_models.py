import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_models_init import *
import numpy as np
from eeg_models_layers import * 

# =============================================================================
#
# =============================================================================

# =============================================================================

class Seq_C1D(nn.Module):
    """
    1D convolutional network with projection of input channels.
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
        
        outputlayer_params['n_classes']=number of output classes
        
        Revised :       05-08-21
    """
    def __init__(self, projlayer_params,convnet_params,n_outputlayer_params):
        super().__init__()

        print('Running class ', self.__class__.__name__)
        
        # Net Parameters
        self.projlayer_params=projlayer_params
        self.convnet_params=convnet_params
        self.outputlayer_params=outputlayer_params
       
        # Projection Layer
        n_channels=projlayer_params['n_channels']
        NneuronsInput= projlayer_params['Nneurons']
        kernel_size=projlayer_params['kernel_size']
        proj_type=projlayer_params['proj_type']
        
        proj_layer= InputChannelProj(
                proj_type=proj_type,n_channels, 
                n_output_loc=NneuronsInput,kernel_size=kernel_size)
        
        # Convolutional  Blocks
        self.ConvBlockList=[]
        Nneurons_Block=convnet_params['Nneurons']
        Kernel_Block=convnet_params['kernel_size']
        
        for Nneurons_conv,kernel_size_conv in zip(Nneurons_Block,Kernel_Block):
            block=Conv1D_StandardBlock(NneuronsInput, Nneurons_conv.copy(),kernel_size_conv.copy(), 
                     batch_config=None,p_drop_loc=0.0)
            self.ConvBlockList.append(block)

        n_features= 16 # change here
        self.features = nn.Sequential(

            nn.Conv1d(n_electrodes, n_features, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv1d(n_features, n_features * 2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv1d(n_features * 2, n_features * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            )

        n_classes=
        self.fc_out = nn.Sequential(
            nn.Linear(n_features * 4, n_classes),
        )

        # weight init
        init_weights_xavier_normal(self)
        
    def parse_projlayer_params(self):
        if 'Nneurons' not in self.projlayer_params.keys():
            self.projlayer_params['Nneurons']=1
        if 'kernel_size' not in self.projlayer_params.keys():
            self.projlayer_params['kernel_size']=1  
        
    def get_embedding(self, x):
        x = self.features(x)
        kernel_size = x.shape[2]
        x = F.avg_pool1d(x, kernel_size) # [N, C, 1]
        x = x.view(x.size(0), -1) # [N, C]
        return x

    def forward(self, x):
        # projection of channels: input x is (NSamp,NChanels,SignalLength)
        x = self.proj_layer(x) # output x is (NSamp,NNeuronsInputLayer,SignalLength)

        # to become invariant to time length
        kernel_size = x.shape[2]
        x = F.avg_pool1d(x, kernel_size) # [N, Nneurons, 1]
        x = x.view(x.size(0), -1) # [N, Nneurons]

        x = self.fc_out(x)
        return x

# =============================================================================

class Seq_C1D_NoStride(nn.Module):
    """
    This model does not perform any pooling or stride in CNN, but performs better
    than the initial old model 15/06/21
    """

    def __init__(self, n_electrodes=14, n_classes=3):
        super().__init__()

        print('Running class ', self.__class__.__name__)

        n_features= 32 # change here
        self.features = nn.Sequential(
            # layer 1
            nn.Conv1d(n_electrodes, n_features, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),

            # layer 2
            nn.Conv1d(n_features, n_features * 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            # layer 3
            nn.Conv1d(n_features * 2,n_features * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )

        self.fc_out = nn.Sequential(
            nn.Linear(n_features * 4, n_classes),
        )

        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):
        x = self.features(x)

        kernel_size = x.shape[2]
        x = F.avg_pool1d(x, kernel_size) # [N, C, 1]
        x = x.view(x.size(0), -1) # [N, C]
        x = self.fc_out(x)
        return x


# =============================================================================
class Seq_C1D_Ensemble(nn.Module):
    """
    Ensemble 1D - CNN

        Three layers:   16 - 32 - 64 layers
        Kernel size:    7, 5, 3, with stride=2, and padding acordingly.

        Revised :       05-08-21
    """
    def __init__(self, n_electrodes=14,  n_classes=2, sel_nodes=None):
        super().__init__()

        in_features = 1
        n_features = 16

        self.sel_nodes = sel_nodes
        self.id_electrodes = np.arange(n_electrodes)
        if sel_nodes is not None:
            self.id_electrodes = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13] # hand pre-defined

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
        if self.sel_nodes is not None:
            x = x[:, self.id_electrodes,:] # get important nodes
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



# =============================================================================
#  spatial temporal models
# =============================================================================

class Seq_C2D_SpatTemp_v1(nn.Module):
    """

    Using 2D CNN - Convolution across time and then accross channels 

    """
    def __init__(self, n_nodes=14, n_classes=2):
        super().__init__()

        print('\tRunning class: ', self.__class__.__name__)

        n_features= 16

        self.temp1 = nn.Sequential(
            nn.Conv2d(1, n_features, kernel_size=(1,3), stride=(1,2)),
            nn.ReLU(inplace=True),
            )

        self.temp2 = nn.Sequential(
            nn.Conv2d(n_features, n_features * 2, kernel_size=(1,3), stride=(1,2)),
            nn.ReLU(inplace=True),
            )

        self.temp3 = nn.Sequential(
            nn.Conv2d(n_features * 2, n_features * 2, kernel_size=(1,3), stride=(1,2)),
            nn.ReLU(inplace=True),
            )
        #n_features * 4 Average of temp3 outputs (n_features*2,14,signal_length)
        self.spat1 = nn.Sequential(
            nn.Conv2d(n_features * 2, n_features * 4, kernel_size=(1,1), stride=(1,1)),
            nn.ReLU(inplace=True),
            )
       #projection of the 14 electrods output using convolution
        self.spat2 = nn.Sequential(
            nn.Conv2d(n_features * 4, n_features * 4, kernel_size=(14,1), stride=(1,1)),
            nn.ReLU(inplace=True),
            )

        self.fc_out = nn.Linear(n_features * 4 * 4, n_classes)

        # weight init
        # init_weights_kaiming_normal(self)
        init_weights_xavier_normal(self)

    def forward(self, x):

        # input is (N, 14, 40)
        x = torch.unsqueeze(x, dim=1) # (N, 1, 14, 40)

        # temporal features
        x = self.temp1(x) # [N, 16, 14, 20]
        x = self.temp2(x) # [N, 32, 14,10]=[NSamp,Nneurons,Nelectrodes,SignalLength]
        x = self.temp3(x) # [N, 32, 14,4]

        # electrode channel features
        x = self.spat1(x) # [N, 64, 14, 4]
        x = self.spat2(x) # [N, 64, 1, 4]

        # to become invariant to input length
        kernel_dim = 4
        x = F.adaptive_avg_pool2d(x, (1, kernel_dim)) # [N, 64, NAvgWeighted]

        # flattening
        x = x.view(x.size(0), -1) # [N, 128]=[N,n_features*NAvgWeighted]
        x = self.fc_out(x)
        return x


class Seq_C2D_SpatTemp_v2(nn.Module):
    """

    Using 2D CNN - Projection of channels and then convolution of temporal projected signals

    """
    def __init__(self, n_nodes=14, n_classes=2):
        super().__init__()

        print('\tRunning class: ', self.__class__.__name__)

        n_features= 16

        #projection of the 14 electrodes input using weighted average
        self.spatial = nn.Sequential(
            nn.Conv2d(1, n_features, kernel_size=(14,1), stride=(1,1)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(n_features),
            )

        self.temp1 = nn.Sequential(
            nn.Conv2d(n_features, n_features, kernel_size=(1,3), stride=(1,2)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(n_features),
            )

        self.temp2 = nn.Sequential(
            nn.Conv2d(n_features, n_features * 2, kernel_size=(1,3), stride=(1,2)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(n_features * 2),
            )

        self.temp3 = nn.Sequential(
            nn.Conv2d(n_features * 2, n_features * 4, kernel_size=(1,3), stride=(1,2)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(n_features * 4),
            )

        self.fc_out = nn.Linear(n_features * 4 * 4, n_classes)

        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):

        # input is (N, 14, 40)
        x = torch.unsqueeze(x, dim=1) # (N, 1, 14, 40)

        # spatial features
        x = self.spatial(x) # [N, 16, 1, 40]

        # temporal features
        x = self.temp1(x) # [N, 16, 1, 19]
        x = self.temp2(x) # [N, 32, 1, 9]
        x = self.temp3(x) # [N, 32, 1, 4]

        # to become invariant to input length
        kernel_dim = 4
        x = F.adaptive_avg_pool2d(x, (1, kernel_dim)) #F.adaptive_avg_pool1d(x.view(x.size(0), -1),kernel_dim)

        # flattening
        x = x.view(x.size(0), -1) # [N, 256]
        x = self.fc_out(x)
        return x


# =============================================================================
# CNN 2D
# =============================================================================


class Seq_C2D(nn.Module):

    """
    Architecture : Implementation of the paper

        EEG-Based Spatio–Temporal Convolutional Neural Network for Driver Fatigue Evaluation
        Gao et al. 2019

    Input data is [N, features=14, timestep=40]
    """

    def __init__(self, n_nodes=14, n_classes=2):
        super().__init__()

        print('running class ', self.__class__.__name__)

        # n_features= 32 # --> 1792 --64.47% | aprende la clase mas numerosa
        n_features= 16 # --> 896   --64%     | trata de compensar entre clases

        ker_temp = (1,3) # kernel for temporal dim
        pad_temp = (0,1) # padding for temporal dim

        self.maxpool_temp = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.avg_pool_kernel_stride = 2

        self.block1 = nn.Sequential(
            nn.Conv2d(1, n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.ReLU(inplace=True),

            # nn.Conv2d(n_features, n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(n_features, n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            )

        self.block2 = nn.Sequential(
            nn.Conv2d(n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.ReLU(inplace=True),

            # nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            )

        self.block3 = nn.Sequential(
            nn.Conv2d(2 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.ReLU(inplace=True),

            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            )

        self.fc = nn.Sequential(
            # input 4480, si n_features=16, kernel_stride = 5
            # input 1792, si n_features=16, kernel_stride = 2
            # input 896, si n_features=8, kernel_stride = 2
            nn.Linear(4 * n_features * 14 * self.avg_pool_kernel_stride, 50),
            nn.ReLU(inplace=True),

            )
        self.fc_out = nn.Linear(50, n_classes)

        # weight init
        init_weights_xavier_normal(self)

    def forward(self, x):

        x = torch.unsqueeze(x, dim=1) # (N, 1, n_nodes, time_step)

        # temporal features
        x = self.maxpool_temp(self.block1(x))
        x = self.maxpool_temp(self.block2(x))
        x = self.block3(x) # [16, 64, 14, 10]

        # to become invariant to input length
        temp_dim = x.shape[-1] # 10
        # if kernel_stride= 5, [N, C, E, 5]
        # if kernel_stride= 2, [N, C, E, 2]
        x = F.avg_pool2d(x, kernel_size=(1, temp_dim // self.avg_pool_kernel_stride),
                         stride=(1, temp_dim // self.avg_pool_kernel_stride))

        # spatial features
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc_out(x)

        return x


class Seq_C2D_TS(nn.Module):
    """
    Input data is [N, features=14, timestep=40]

    The sequence of processing:  1. Temporal features, next 2)  Spatial features

    Architecture : Adapted from the paper

        Continuous EEG Decoding of Pilots’ Mental States Using Multiple Feature Block-Based Convolutional Neural Network
        Lee et al. 2020

    """
    def __init__(self, n_nodes=14, n_classes=2):
        super().__init__()

        print('\tRunning class: ', self.__class__.__name__)

        self.dropout = nn.Dropout2d(p=0.2)
        n_features= 16

        ker_temp = (1,3) # kernel for temporal dim
        pad_temp = (0,1) # padding for temporal dim

        k_spat = (3,1) # kernel for spatial dim
        p_spat = (1,0) # padding for spatial dim

        self.avg_pool_kernel_stride = 5

        self.maxpool_temp = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

        self.maxpool_spat = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.avgpool_spat = nn.AvgPool2d(kernel_size=(7,1), stride=(7,1))

        self.block_s1 = nn.Sequential(
            nn.Conv2d(1, n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.ReLU(inplace=True),
            # nn.Conv2d(n_features, n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(n_features, n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            )

        self.block_s2 = nn.Sequential(
            nn.Conv2d(n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.ReLU(inplace=True),
            # nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            )

        self.block_s3 = nn.Sequential(
            nn.Conv2d(2 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.ReLU(inplace=True),
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            )


        self.block_t1 = nn.Sequential(
            nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=k_spat, padding=p_spat),
            nn.ReLU(inplace=True),
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=k_spat, padding=p_spat),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=k_spat, padding=p_spat),
            # nn.ReLU(inplace=True),
            )

        self.block_t2 = nn.Sequential(
            nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=k_spat, padding=p_spat),
            nn.ReLU(inplace=True),
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=k_spat, padding=p_spat),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=k_spat, padding=p_spat),
            # nn.ReLU(inplace=True),
            )

        # full connected
        self.fc = nn.Sequential(
            nn.Linear(320, 64),
            nn.ReLU(inplace=True),
            )

        self.fc_out = nn.Linear(64, n_classes)

        # weight init
        # 7/3/21 kaiming_normal in hour experiment is better than xavier_normal
        # init_weights_kaiming_normal(self)
        init_weights_xavier_normal(self)


    def forward(self, x):

        x = torch.unsqueeze(x, dim=1) # (N, 1, n_nodes, time_step)

        # temporal features
        x = self.maxpool_temp(self.block_s1(x))
        x = self.maxpool_temp(self.block_s2(x))
        x = self.block_s3(x)

        # to become invariante to vector length
        # the outputs is always [N, C, E, T] , T = length / 4
        temp_dim = x.shape[-1]
        x = F.avg_pool2d(x, kernel_size=(1, temp_dim // self.avg_pool_kernel_stride),
                         stride=(1, temp_dim // self.avg_pool_kernel_stride)) # [N, C, E, 5]

        # spatial features
        x = self.maxpool_spat(self.block_t1(x)) # [N, 64, 7, 5]
        x = self.avgpool_spat(self.block_t2(x)) # [N, 64, 1, 5]


        x = x.view(x.size(0), -1) # [320]
        x = self.fc(x) # [64]
        x = self.fc_out(x)
        return x


class Seq_C2D_ST(nn.Module):
    """
    Input data is [N, features=14, timestep=40]

     The sequence of processing:  1. Spatial features, next 2) Temporal features

    Architecture : Adapted from the paper

        Continuous EEG Decoding of Pilots’ Mental States Using Multiple Feature Block-Based Convolutional Neural Network
        Lee et al. 2020

    """
    def __init__(self, n_nodes=14, n_classes=2):
        super().__init__()

        print('\tRunning class: ', self.__class__.__name__)

        # self.dropout = nn.Dropout2d(p=0.2)
        n_features= 16

        ker_temp = (1,3) # kernel for temporal dim
        pad_temp = (0,1) # padding for temporal dim

        k_spat = (3,1) # kernel for spatial dim
        p_spat = (1,0) # padding for spatial dim

        self.avg_pool_kernel_stride = 5

        self.maxpool_temp = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))

        self.maxpool_spat = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        # self.avgpool_spat = nn.AvgPool2d(kernel_size=(7,1), stride=(7,1))

        self.block_s1 = nn.Sequential(
            nn.Conv2d(1, n_features, kernel_size=k_spat, padding=p_spat),
            nn.ReLU(inplace=True),
            # nn.Conv2d(n_features, n_features, kernel_size=k_spat, padding=p_spat),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(n_features, n_features, kernel_size=k_spat, padding=p_spat),
            # nn.ReLU(inplace=True),
            )

        self.block_s2 = nn.Sequential(
            nn.Conv2d(n_features, 2 * n_features, kernel_size=k_spat, padding=p_spat),
            nn.ReLU(inplace=True),
            # nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=k_spat, padding=p_spat),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=k_spat, padding=p_spat),
            # nn.ReLU(inplace=True),
            )

        self.block_t1 = nn.Sequential(
            nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.ReLU(inplace=True),
            # nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(2 * n_features, 2 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            )

        self.block_t2 = nn.Sequential(
            nn.Conv2d(2 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.ReLU(inplace=True),
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            )

        self.block_t3 = nn.Sequential(
            nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            nn.ReLU(inplace=True),
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(4 * n_features, 4 * n_features, kernel_size=ker_temp, padding=pad_temp),
            # nn.ReLU(inplace=True),
            )

        # full connected
        self.fc = nn.Sequential(
            nn.Linear(960, 100),
            nn.ReLU(inplace=True),
            )

        self.fc_out = nn.Linear(100, n_classes)

        # weight init
        # 7/3/21 kaiming_normal in hour experiment is better than xavier_normal
        # init_weights_kaiming_normal(self)
        init_weights_xavier_normal(self)


    def forward(self, x):

        x = torch.unsqueeze(x, dim=1) # (N, 1, n_nodes, time_step)

        # spatial features
        x = self.maxpool_spat(self.block_s1(x)) # [N, 16, 7, 40]
        # x = self.avgpool_spat(self.block_s2(x)) # [N, 32, 1, 40]
        x = self.maxpool_spat(self.block_s2(x)) # [N, 32, 3, 40]

        # temporal features
        x = self.maxpool_temp(self.block_t1(x)) # [N, 32, 3, 20]
        x = self.maxpool_temp(self.block_t2(x)) # [N, 64, 3, 10]
        x = self.block_t3(x)

        # to become invariant to input length
        # # the outputs is always [N, C, E, T] , T = length / 4
        temp_dim = x.shape[-1]
        x = F.avg_pool2d(x, kernel_size=(1, temp_dim // self.avg_pool_kernel_stride),
                         stride=(1, temp_dim // self.avg_pool_kernel_stride)) # [N, 64, 3, 5]

        x = x.view(x.size(0), -1) # [960]
        x = self.fc(x) # [128]
        x = self.fc_out(x)
        return x



# =============================================================================
# =============================================================================
class My_ResNetBlock(nn.Module):

    def __init__(self,input_channels, output_channels, stride=1, use_1x1conv=True):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(output_channels)

        # self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm1d(output_channels)

        self.conv3 = nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm1d(output_channels)

        if use_1x1conv:
            self.conv_1x1 = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm1d(output_channels),  # this help ?
                )
        else:
            self.conv_1x1 = None

    def forward(self, x):

        # out = self.bn1(self.conv1(x))
        out = self.conv1(x)
        out = F.relu(out)

        # out = self.bn2(self.conv2(out))
        # out = F.relu(out)

        # out = self.bn3(self.conv3(out))
        out = self.conv3(out)

        if self.conv_1x1:
            x = self.conv_1x1(x)
        out += x
        return F.relu(out)


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

# resnet 3 blocks of 3 cnn
# agusti: 0.89 (+/- 0.016)
# alejandro: 0.77 (+/- 0.017)
# cristian: 0.71 (+/- 0.06)
# dani: 0.74 (+/- 0.031)
# jose: 0.70 (+/- 0.031)
# qiang: 0.91 (+/- 0.034)
# Average Accuracy 0.79














# Incrementar el kernel empeora el resultado (k=7, 9, 11)
# baseline for CNN
# 53 % in CRISTIAN.
# self.feats = nn.Sequential(
#             nn.Conv1d(1, n_filters, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm1d(n_filters),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=5, stride=4),

#             nn.Conv1d(n_filters, n_filters * 2, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm1d(n_filters * 2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=5, stride=4),

#             nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=3, stride=1, padding=0),
#             nn.BatchNorm1d(n_filters * 4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=5, stride=4),

#             nn.MaxPool1d(kernel_size=7, stride=1)
#             )
# self.classifier = nn.Sequential(
#             nn.Linear(64 , 3)
#             )

# NOTAS: test a epocas 50

# Exp 1: la salida de las convoluciones, se classifica directamente, alcanzando 59%
# (a 100 epocas alcanza 64%)
# self.classifier = nn.Sequential(
#             nn.Linear(512 , 3)
#             )
# El loss del VAL es menor que el TRAIN, hasta la epoca 60 apox donde chocan, el val loss
# se hace constante, mientras que el train loss decrese . NO hay overfitting.


# Exp 2: a la salida de las convoluciones agregar avgpooling (64, pues se ha utilizado al final average 8)
# se obtiene 56%. El loss del VAL es menor que el TRAIN
# y van separados. NO hay overfitting.

# Exp 3: Como el average final perjudica, agregamos una capa adicional antes del classifier

# self.features = nn.Sequential(
        #     nn.Linear(64 * 8, 64),
        #     )

# se alcanza 62%, los losses van juntos hasta la epoca 50 aprox. en la epoca 100 compienza a subir

# NOTA GENERAL maxpooling utilizan PADDING=1, si colocamos a 0, cae a 2% aprox. en todos los experimentos
# asimismo, utilizar avg pooling despues de las convoluciones perjudica en 2% aprox.

# Experimentos con LSTM

# 64% 150 epochs: 3 niveles CNN (64-dim feats), lstm in=69 de 100 cell.
#   A partir epocas 100 loss creciente, overfitting
# 65% 150 epochs: 3 niveles CNN (64-dim feats), lstm in=69 de 50 cell.
#   hasta 150 epocas loss constante, no overfitting

# 61% 150 epochs: 3 niveles CNN (128-dim feats), lstm in=69 de 100 cell.
#   hasta 150 epocas loss constante, no overfitting
# 64% 150 epochs: 3 niveles CNN (128-dim feats), lstm in=69 de 50 cell.
#   hasta 100 epocas loss constante, no overfitting


# 67% 150 epochs: 4 niveles CNN (128-dim feats), lstm in=34 de 100 cell.
#   A partir epoca 100 loss creciente
# 64% 150 epochs: 4 niveles CNN (128-dim feats), lstm in=34 de 50 cell.
#   A partir epoca 130 loss creciente

# 65% 150 epochs: 4 niveles CNN (128-dim feats), lstm in=128 de 100 cell.
#   A partir epoca 80 loss creciente
# 64% 150 epochs: 4 niveles CNN (128-dim feats), lstm in=128 de 50 cell.
#   A partir epoca 100 loss creciente
