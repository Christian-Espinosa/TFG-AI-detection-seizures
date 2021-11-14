# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:00:28 2021

@author: debora
"""

import time
import numpy as np
from sklearn import metrics

import torch

import matplotlib.pyplot as plt




def plot_metrics(avg_cost, msg= None):

    if msg is None:
        msg = ''

    train_loss = avg_cost[:,0]
    valid_loss = avg_cost[:,3]

    train_acc = avg_cost[:,1]
    valid_acc = avg_cost[:,4]

    plt.plot(train_loss, label='training loss')
    plt.plot(valid_loss, label='validation loss')
    plt.legend()
    plt.xlabel(r'epochs')
    plt.title(msg + ' Training Loss ')
    plt.show()

    plt.plot(train_acc, label='training')
    plt.plot(valid_acc, label='validation')
    plt.legend()
    plt.xlabel(r'epochs')
    plt.title(msg + ' Accuracy ' )
    plt.show()

# saving and loading checkpoint mechanisms
def save_checkpoint(model, optimizer, val_loss, epoch, save_full_path):
    """
        Save a model and its weights.
    """
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss,
                  'epoch': epoch}
    torch.save(state_dict, save_full_path)
    print(f'\tSaved model at epoch {epoch}  ==> {save_full_path}')

def load_checkpoint(model, optimizer, save_full_path):
    """
        Load a model and its weights.
    """
    state_dict = torch.load(save_full_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_loss = state_dict['val_loss']
    epoch = state_dict['epoch']
    print(f'Loaded model weights <== {save_full_path}')
    return epoch, val_loss

# saving and loading checkpoint mechanisms
def save_full_checkpoint(model, optimizer, save_full_path, metadata=None):
    """
        Save a model and its weights.
    """
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'metadata': metadata}
    torch.save(state_dict, save_full_path)
    print(f'\tSaved model weights at  ==> {save_full_path}')

def load_full_checkpoint(model, optimizer, save_full_path):
    """
        Load a model and its weights.
    """
    state_dict = torch.load(save_full_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    metadata = state_dict['metadata']
    print(f'Loaded model <== {save_full_path}')
    return metadata


