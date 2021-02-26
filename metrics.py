#!/usr/bin/env python
# coding: utf-8
import torch
from segmentation_models_pytorch.utils import base
from sklearn import metrics
from segmentation_models_pytorch.utils.base import Activation
from skimage.metrics import hausdorff_distance

class Haussdorff(base.Metric):
    __name__ = 'haussdorff'

    def __init__(self, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, predictions, labels):
        predictions = self.activation(predictions)
        dh = torch.tensor(hausdorff_distance(predictions.numpy(), labels.numpy()))
        return dh


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