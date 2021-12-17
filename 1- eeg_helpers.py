# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:30:28 2020

@author: jyauri
"""
import time
import numpy as np
from sklearn import metrics

import torch
import torch.nn as nn

from eeg_utilitarios import *


def train_model(model, optimizer, criterion, train_dataloader, valid_dataloader,
                n_epochs, verbose=1, save_path='./pretrained/model.pt',
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
            # outputs = model(seqs)
            outputs, _ = model(seqs) # returned second value is probs
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            cost[0] = loss.item()
            cost[1], cost[2], _ = compute_metrics(outputs, targets)
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
                    # outputs = model(seqs)
                    outputs, _ = model(seqs) # returned second value is probs
                    loss = criterion(outputs, targets)

                    cost[3] = loss.item()
                    cost[4], cost[5], _ = compute_metrics(outputs, targets)
                    avg_cost[index, 3:] += cost[3:] / total_valid_batch

            # if avg_cost[index, 3] < best_val_loss:
            #     best_val_loss = avg_cost[index, 3]
            #     print('model saved at epoch ', index+1)
            #     save_checkpoint(save_path, model, optimizer, best_val_loss)

        if verbose:
            print(f'Epoch [{epoch + 1}/{n_epochs}] | TRAIN: Loss:{avg_cost[index,0]:.2f} Acc:{avg_cost[index,1]:.2f} Pre:{avg_cost[index,2]:.2f} |' +
              f' TEST: Loss:{avg_cost[index,3]:.2f} Acc:{avg_cost[index,4]:.2f} Pre:{avg_cost[index,5]:.2f}')

    time_elapsed = time.time() - time_start
    display_elapsed_time(time_elapsed)
    return model, avg_cost


def test_model(model, test_dataloader):

    total_test_batch = len(test_dataloader)

    y_true = []
    y_pred = []
    y_pred_probs = []
    model.eval()
    with torch.no_grad():
        iter_test_dataset = iter(test_dataloader)
        for k in range(total_test_batch):
            seqs, targets = next(iter_test_dataset)
            seqs, targets = seqs.cuda(), targets.cuda()
            print("seqs: ",seqs)
            outputs, probs = model(seqs)
            pred = outputs.max(1, keepdim=True)[1].cpu().numpy()

            y_pred.extend(list(pred.reshape(-1,)))
            y_pred_probs.append(probs.cpu().numpy())
            y_true.extend(list(targets.cpu().numpy()))

    # avg_acc = metrics.accuracy_score(y_true, y_pred)
    # avg_prec = metrics.precision_score(y_true, y_pred,
    #                                     labels=np.unique(y_true),
    #                                     average='weighted',
    #                                     zero_division=1)

    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred_probs = np.round(y_pred_probs, 2)

    return (y_true, y_pred), y_pred_probs

