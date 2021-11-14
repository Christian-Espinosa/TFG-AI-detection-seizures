# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:00:28 2021

@author: debora
"""


import numpy as np
from sklearn import model_selection

import torch
import torch.nn as nn

from train_losses import WeightedCrossEntropy
from DataSets.datasets_util import *

def  TrainPipeLine(train_data_x,train_data_y,
                         MODEL_CONFIG,TRAIN_CONFIG,OPTIMIZER_CONFIG):

        # 1) Preprocessing
        x_train, scaler = scalers_fit(train_data_x)
        y_train = train_data_y.astype(np.int64)
    
        # 2). train model 
        
        # 2.1). train-validation sets    
        x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
            x_train, y_train,
            test_size=TRAIN_CONFIG['test_size'],                            
            stratify=y_train, random_state=123)
                                            
            
        # 2.2) Model Fit
        model, optimizer, avg_cost = train_model(x_train, y_train, TRAIN_CONFIG, MODEL_CONFIG, 
                    OPTIMIZER_CONFIG, x_valid, y_valid )
        
        return  model, optimizer, avg_cost,scaler
    
def train_model(x_train, y_train, TRAIN_CONFIG, MODEL_CONFIG, 
                OPTIMIZER_CONFIG, x_valid=None, y_valid=None):

    ## Input Parameters
    transf=TRAIN_CONFIG['transf']
    batch_size=TRAIN_CONFIG['batch_size']
    n_epochs=TRAIN_CONFIG['n_epochs']
    pth_full_name= TRAIN_CONFIG['pth_full_name']
    balanced=TRAIN_CONFIG['balanced']
   
    
    # DataLoaders
    if balanced==True:
        train_dataloader = create_dataloader_balanced(x_train, y_train, transf, batch_size)
    else:
        train_dataloader = create_dataloader(x_train, y_train, transf, batch_size)
    
    valid_dataloader = None
    if x_valid is not None and y_valid is not None:
        if balanced==True:
            valid_dataloader = create_dataloader_balanced(x_valid, y_valid, transf, batch_size, shuffle=False)
        else:
            valid_dataloader = create_dataloader(x_valid, y_valid, transf, batch_size, shuffle=False)

      
       

    # Define model if needed
    model=MODEL_CONFIG['model']
    if  model is None:
        model, optimizer, scheduler = instance_model(MODEL_CONFIG['MODEL_NAME'], MODEL_CONFIG,OPTIMIZER_CONFIG)
    
    criterion = WeightedCrossEntropy(y_train) 

    # Train the model
    model, avg_cost = standard_fit(model, optimizer, criterion,
                            train_dataloader, valid_dataloader,
                            n_epochs,
                            verbose=False,
                            save_path=pth_full_name,
                            best_val_loss=None)

    return model, optimizer, avg_cost

def standard_fit(model, optimizer, criterion, train_dataloader, valid_dataloader,
                n_epochs, verbose=1, save_path=None,
                best_val_loss=None):

    if best_val_loss is None:
        best_val_loss = float("Inf")

    avg_cost = np.zeros([n_epochs, 6], dtype=np.float32)
    time_start = time.time()

    # Training the model for TOTAL_EPOCHS
    total_train_batch = len(train_dataloader)
    for epoch in range(n_epochs):
        index = epoch
        cost = np.zeros(6, dtype=np.float32)

        # training
        model.train()
        iter_train_dataset = iter(train_dataloader)
        for k in range(total_train_batch):
            seqs, targets = next(iter_train_dataset)
            seqs, targets = seqs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            cost[0] = loss.item()
         #   cost[1], cost[2], _ = compute_metrics(outputs, targets)
            avg_cost[index, :3] += cost[ :3] / total_train_batch

        # validation
        if valid_dataloader is not None:
            total_valid_batch = len(valid_dataloader)
            model.eval()
            with torch.no_grad():
                iter_valid_dataset = iter(valid_dataloader)
                for k in range(total_valid_batch):
                    seqs, targets = next(iter_valid_dataset)
                    seqs, targets = seqs.cuda(), targets.cuda()
                    outputs = model(seqs)
                    loss = criterion(outputs, targets)

                    cost[3] = loss.item()
                #    cost[4], cost[5], _ = compute_metrics(outputs, targets)
                    avg_cost[index, 3:] += cost[3:] / total_valid_batch

            # save model
            if (save_path is not None) & (avg_cost[index, 3] < best_val_loss):
                best_val_loss = avg_cost[index, 3]
                save_checkpoint(model, optimizer, best_val_loss, epoch + 1, pth_full_name + '_ep_' + str(epoch + 1) + ext)


        if verbose:
            print(f'Epoch [{epoch + 1}/{n_epochs}] | TRAIN: Loss:{avg_cost[index,0]:.2f} Acc:{avg_cost[index,1]:.2f} Pre:{avg_cost[index,2]:.2f} |' +
              f' TEST: Loss:{avg_cost[index,3]:.2f} Acc:{avg_cost[index,4]:.2f} Pre:{avg_cost[index,5]:.2f}')

    time_elapsed = time.time() - time_start
    display_elapsed_time(time_elapsed)
    
        # Try to free GPU memory
    seqs.cpu()
    targets.cpu()
    outputs.cpu()
    loss.cpu()
    del loss
    del outputs
    del seqs
    del targets
    
    return model, avg_cost
