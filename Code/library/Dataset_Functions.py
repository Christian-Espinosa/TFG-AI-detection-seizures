

import os, sys
import numpy as np
import math
import torch
import pandas as pd
import pyarrow.parquet as pq

def CheckModel(model):

    print(model)
    #Revisar que el model funciona correctament

    NSamp=1
    n_channel=2
    L=4
    x_test =torch.randn(NSamp,n_channel,L).cuda()
    print("*********************")
    print(x_test)
    print(x_test.shape)
    y = model(x_test)
    print(y.shape)
    print(y)

    NSamp=10
    n_channel=1
    L=40
    x_test=torch.randn(NSamp,n_channel,L)
    model(x_test)

    NSamp=10
    n_channel=14
    L=60
    x_test=torch.randn(NSamp,n_channel,L)
    model(x_test)

def SplitData(data, perc):
    #define percentage of train
    rows = len(data[0])
    n_tr = int(math.floor(rows*perc))
    train = data.head(n_tr)
    test = data[n_tr:]
    return train, test

def train_model(model, optimizer, criterion, train_dataloader, valid_dataloader, n_epochs):

    # Training the model for TOTAL_EPOCHS
    total_train_batch = len(train_dataloader)
    for epoch in range(n_epochs):

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

    return model

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
