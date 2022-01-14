import os, sys
import numpy as np
import math
import torch
import pandas as pd
import pyarrow.parquet as pq
from sklearn import preprocessing

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

##############Spliting Datasets
def SplitDataPandas(data, perc, labeloffirstelement = 'FP1-F7'):
    #define percentage of train
    rows = len(data[labeloffirstelement])
    n_tr = int(math.floor(rows*perc))
    train = data.iloc[:n_tr]
    test = data.iloc[n_tr:]
    return train, test
    """
    
def select_subject_train_test_data(data_x, data_y, percentage = 0.5):

    rows = data_x.shape[0]
    chosen_rows = int(math.floor(rows*percentage))

    train_data_x = data_x[:chosen_rows, :, :]
    train_data_y = data_y[:chosen_rows, :]
    test_data_x = data_x[chosen_rows:, :, :]
    test_data_y = data_y[chosen_rows:, :]
    return train_data_x, train_data_y, test_data_x, test_data_y
"""
def select_subject_train_test_data(data_x, data_y, percentage = 0.5):

    rows = len(data_x)
    chosen_rows = int(math.floor(rows*percentage))

    train_data_x = data_x[~chosen_rows, :]
    train_data_y = data_y[~chosen_rows, :]
    test_data_x = data_x[chosen_rows, :]
    test_data_y = data_y[chosen_rows, :]
    return train_data_x, train_data_y, test_data_x, test_data_y

##############Normalize
def scalers_fit(x_train):
    # source https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
    scalers = {}
    for i in range(x_train.shape[1]):
        scalers[i] = preprocessing.StandardScaler()
        # scalers[i] = preprocessing.MaxAbsScaler()
        # scalers[i] = preprocessing.MinMaxScaler()
        x_train[:, i, :] = scalers[i].fit_transform(x_train[:, i, :])
    return x_train, scalers

def scalers_transform(scalers, x_test):
    for i in range(x_test.shape[1]):
        x_test[:, i, :] = scalers[i].transform(x_test[:, i, :])
    return x_test


"""
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
"""


def train_model(model, optimizer, criterion, train_dataloader, valid_dataloader,
                n_epochs, verbose=1, save_path='./pretrained/model.pt',
                best_val_loss=None):

    if best_val_loss is None:
        best_val_loss = float("Inf")

    avg_cost = np.zeros([n_epochs, 6], dtype=np.float32)

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
            #cost[1], cost[2], _ = compute_metrics(outputs, targets)
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
                    #cost[4], cost[5], _ = compute_metrics(outputs, targets)
                    avg_cost[index, 3:] += cost[3:] / total_valid_batch

            # if avg_cost[index, 3] < best_val_loss:
            #     best_val_loss = avg_cost[index, 3]
            #     print('model saved at epoch ', index+1)
            #     save_checkpoint(save_path, model, optimizer, best_val_loss)

        if verbose:
            print(f'Epoch [{epoch + 1}/{n_epochs}] | TRAIN: Loss:{avg_cost[index,0]:.2f} Acc:{avg_cost[index,1]:.2f} Pre:{avg_cost[index,2]:.2f} |' +
              f' TEST: Loss:{avg_cost[index,3]:.2f} Acc:{avg_cost[index,4]:.2f} Pre:{avg_cost[index,5]:.2f}')

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