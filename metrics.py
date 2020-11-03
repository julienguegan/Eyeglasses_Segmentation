#!/usr/bin/env python
# coding: utf-8
import torch
from numpy import isnan


def IoU_score(predictions, labels):

    predictions = predictions.detach()
    intersection = torch.logical_and(predictions, labels).float().sum((1, 2))
    union        = torch.logical_or(predictions, labels).float().sum((1, 2))
    # compute IoU 
    iou = (intersection / union)
    # replace nan value by 0
    mask_nan      = isnan(iou.numpy())
    iou[mask_nan] = torch.tensor(float(0))
    # average IoU because there is a batch of image
    
    return iou.mean()