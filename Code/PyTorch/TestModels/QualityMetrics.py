# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:51:37 2021

@author: debora
"""

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize




def confusion_matrix_calculate(y_true, y_pred, tags_categ=None):

    if tags_categ is None:
        tags_categ = list(set(y_true))

    tags_ord = np.arange(len(tags_categ))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=tags_ord, normalize='true')
    return cm

def ClassificationMetrics(y_true, y_pred, tags_categ=None):
    
    y_true=y_true.astype('int64')
    y_pred=y_pred.astype('int64')
    c_m = confusion_matrix_calculate(y_true, y_pred, tags_categ)
    prec,rec,_,_ = metrics.precision_recall_fscore_support(y_true, y_pred,
                                       zero_division=0)
    
    return prec,rec,c_m

def AUCMetrics(y_true, y_prob):
    
     n_classes=y_prob.shape[1]
     NSamp=y_prob.shape[0]
     y_gt = label_binarize(np.array(y_true).astype(int), classes=np.arange(n_classes))
     y_true=np.array(y_true).astype(int)

     if n_classes==2:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
     else:
         fpr=[]
         tpr=[]
         thresholds=[]
         roc_auc=np.empty_like(y_gt)
         for i in range(y_gt.shape[1]):
             fpraux, tpraux, thresholdsaux= metrics.roc_curve(y_gt[:, i], y_prob[:, i])
             fpr.append(fpraux)
             tpr.append(tpraux)
             thresholds.append(thresholdsaux)
             roc_auc = metrics.auc(fpraux, tpraux)
    
     return roc_auc,fpr,tpr,thresholds