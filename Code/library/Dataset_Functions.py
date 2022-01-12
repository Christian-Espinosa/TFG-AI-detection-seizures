

import os, sys
import numpy as np
import math
import torch
import pandas as pd
import pyarrow.parquet as pq

def CheckModel(model, device):

    model.to(device)
    print(model)
    #Revisar que el model funciona correctament

    n_window=4
    n_channel=22
    L=40*256
    x_test =torch.randn(n_window,n_channel,L).to(device)
    print("*********************")
    #print(x_test)
    #print(x_test.shape)
    y = model(x_test)
    print(y.shape)
    print(y)


def SplitData(data, perc, labeloffirstelement = 'FP1-F7'):
    #define percentage of train
    rows = len(data[labeloffirstelement])
    n_tr = int(math.floor(rows*perc))
    train = data.iloc[:n_tr]
    test = data.iloc[n_tr:]
    return train, test

def train_model(model, optimizer, criterion, train_dataloader, valid_dataloader,
                n_epochs, verbose=1):

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
"""
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
"""
    

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

    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred_probs = np.round(y_pred_probs, 2)

    return (y_true, y_pred), y_pred_probs
