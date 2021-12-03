import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
#import torchvision

import numpy as np

import matplotlib.pyplot as plt

# =============================================================================
# Custom dataset
# =============================================================================
class EEG_Dataset(data.Dataset):
    def __init__(self, X, Y, transformation=None):
        super().__init__()
        self.X = X
        self.y = Y
        self.transformation = transformation

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.from_numpy(np.array(self.y[idx])).long()

# =============================================================================
# Transformation
# =============================================================================

class IdentitySeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):
        return x  # [n_feats, time_steps]

class ReverseSeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):
        x = np.ascontiguousarray(x[:, ::-1])
        return x # [n_feats, time_steps]

class ToTensorSeries(object):
    """
    x is numpy dtype
    """
    def __call__(self, x):

        return torch.from_numpy(x) # [n_feats, time_steps]

# if __name__ == "__main__":

#     trans = torchvision.transforms.Compose([
#         IdentitySeries(),
#         ReverseSeries(),
#         ToTensorSeries(),
#         ])

    # out = ReverseSeries()(x)
        # out = trans(x)

    # plot
    # fig , ax = plt.subplots(14,1)
    # for i in range(14):
    #     data_y = x[i,:]
    #     data_x = np.arange(1, data_y.size(0)+1) / 8.0
    #     ax[i].plot(data_x, data_y)
    #     ax[i].axes.yaxis.set_visible(False)
    # fig.tight_layout()