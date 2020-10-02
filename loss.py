#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super().__init__()
        self.thresh    = -torch.log(torch.tensor(thresh, dtype=torch.float)).to(device)
        self.n_min     = n_min
        self.ignore_lb = ignore_lb
        self.criteria  = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss       = self.criteria(logits, labels).view(-1)
        loss, _    = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores    = F.softmax(logits, dim=1)
        factor    = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss      = self.nll(log_score, labels)
        return loss

class BCELoss2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predict, target):
        predict = predict.view(-1)
        target  = target.view(-1)
        return self.bce_loss(predict, target)

class NLLLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, reduction='mean') #NLLLoss2d

    def forward(self, predict, target):
        return self.nll_loss(F.log_softmax(predict, dim=-1), target)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight, size_average, reduction='mean') #NLLLoss2d

    def forward(self, predict, target):
        predict = predict.view(-1)
        target  = target.view(-1)
        return self.ce_loss(predict, target)
    
class DiceLoss(nn.Module):

    def __init__(self, eps=1., beta=1., ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.ignore_channels = ignore_channels

    def forward(self, predict, target):
        
        return 1 - f_score(predict, target, beta=self.beta, eps=self.eps, threshold=None, ignore_channels=self.ignore_channels )
    
def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x
    
    