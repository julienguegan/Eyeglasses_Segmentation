#!/usr/bin/env python
# coding: utf-8
import torch
from numpy import isnan
from scipy.spatial.distance import directed_hausdorff
from segmentation_models_pytorch.utils import base
from sklearn import metrics

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

def uniq(a):
    return set(torch.unique(a.cpu()).numpy())

def sset(a, sub):
    return uniq(a).issubset(sub)

def simplex(t, axis=1):
    _sum  = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t, axis=1):
    return simplex(t, axis) and sset(t, [0, 1])

def haussdorf(preds, target):
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)
    B, C, _, _ = preds.shape
    res        = torch.zeros((B, C), dtype=torch.float32, device=preds.device)
    n_pred     = preds.cpu().numpy()
    n_target   = target.cpu().numpy()

    for b in range(B):
        if C == 2:
            res[b, :] = numpy_haussdorf(n_pred[b, 0], n_target[b, 0])
            continue

        for c in range(C):
            res[b, c] = numpy_haussdorf(n_pred[b, c], n_target[b, c])

    return res

def numpy_haussdorf(pred, target):
    assert len(pred.shape) == 2
    assert pred.shape == target.shape
    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def confusion_matrix(pred, target): 
    tn, fp, fn, tp = metrics.confusion_matrix(pred.view(-1), target.view(-1)).ravel()
    return tn, fp, fn, tp 
    
    
class ConfusionMatrix(base.Metric):
    __name__ = 'confusion_matrix'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return metrics.confusion_matrix(y_pr.view(-1), y_gt.view(-1))