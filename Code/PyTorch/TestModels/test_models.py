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




def eval_model(model, test_dataloader):
    
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
            outputs = model(seqs)
            probs = torch.softmax(outputs, dim=1)
            pred = outputs.max(1, keepdim=True)[1].cpu().numpy()

            y_pred.extend(list(pred.reshape(-1,)))
            y_pred_probs.append(probs.cpu().numpy())
            y_true.extend(list(targets.cpu().numpy()))

    y_pred_probs = np.concatenate(y_pred_probs, axis=0)
    y_pred_probs = np.round(y_pred_probs, 2)

    return y_true, y_pred, y_pred_probs



