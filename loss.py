#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils.base import Activation
from skimage.metrics import hausdorff_distance
import numpy as np 
from torch import einsum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OhemBCELoss(base.Loss):
    def __init__(self, thresh=0.7, n_min=6, ignore_lb=255, *args, **kwargs):
        super().__init__(**kwargs)
        self.thresh    = -torch.log(torch.tensor(thresh, dtype=torch.float)).to(device)
        self.n_min     = n_min
        self.ignore_lb = ignore_lb
        self.criteria  = nn.BCELoss()

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss       = self.criteria(logits, labels).view(-1)
        loss, _    = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(base.Loss):
    def __init__(self, gamma=2, ignore_lb=255, *args, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.nll   = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores    = F.softmax(logits, dim=1)
        factor    = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss      = self.nll(log_score, labels)
        return loss
    
    
   
    
def jaccard(intersection, union, eps=1e-15):
    return (intersection) / (union - intersection + eps)

def dice(intersection, union, eps=1e-15, smooth=1.):
    return (2. * intersection + smooth) / (union + smooth + eps)


class BCEDiceLoss(base.Loss):
    
    def __init__(self, bce_weight=0.3, mode="dice", eps=1e-7, weight=None, smooth=1., **kwargs):
        super().__init__(**kwargs)
        self.nll_loss   = torch.nn.BCEWithLogitsLoss(weight=weight)
        self.bce_weight = bce_weight
        self.eps        = eps
        self.mode       = mode
        self.smooth     = smooth

    def forward(self, outputs, targets):    
        loss = self.bce_weight * self.nll_loss(outputs, targets)

        if self.bce_weight < 1.:
            targets      = (targets == 1).float()
            outputs      = torch.sigmoid(outputs)
            intersection = (outputs * targets).sum()
            union        = outputs.sum() + targets.sum()
            if self.mode == "dice":
                score = dice(intersection, union, self.eps, self.smooth)
            elif self.mode == "jaccard":
                score = jaccard(intersection, union, self.eps)
            loss -= (1 - self.bce_weight) * torch.log(score)
        return loss
    
    
    
class BinaryFocalLoss(base.Loss):
    """ https://arxiv.org/abs/1708.02002)
    inputs :
        - alpha: (tensor) 3D or 4D the scalar factor for this criterion
        - gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more focus on hard misclassified example
        - reduction: `none`|`mean`|`sum`
        - **kwargs
            balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=[1.0, 1.0], gamma=2, ignore_index=None, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha        = alpha
        self.gamma        = gamma
        self.smooth       = 1e-6
        self.ignore_index = ignore_index
        self.reduction    = reduction
        assert self.reduction in ['none', 'mean', 'sum']
        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert self.alpha.shape[0] == 2, 'the `alpha` shape is not match the number of class'
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)
        else:
            raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.clamp(output, self.smooth, 1.0 - self.smooth)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        pos_loss = -self.alpha[0] * torch.pow(torch.sub(1.0, prob), self.gamma) * torch.log(prob) * pos_mask
        neg_loss = -self.alpha[1] * torch.pow(prob, self.gamma) * torch.log(torch.sub(1.0, prob)) * neg_mask
        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos  = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg  = neg_mask.view(neg_mask.size(0), -1).sum()
        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss    
    
class SurfaceLoss(base.Loss):
    '''https://github.com/LIVIAETS/boundary-loss'''
    def __init__(self, idx_filtered=1):
        super(SurfaceLoss, self).__init__()
        # idx_filtered is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = idx_filtered #kwargs["idc"]

    def forward(self, probs, dist_maps):

        pc = probs[:, ...]#self.idc, ...].type(torch.float32)
        dc = dist_maps[:, ...]#self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss
    
class GeneralizedDice(base.Loss):
    def __init__(self, idx_filtered=1):
        super(GeneralizedDice, self).__init__()
        # idx_filtered is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = idx_filtered

    def forward(self, probs, target):

        pc = probs[:, ...]
        tc = target[:, ...]

        w = 1 / ((einsum("bcwh->bc", tc) + 1e-10) ** 2)
        intersection = w * einsum("bcwh,bcwh->bc", pc, tc)
        union = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss    
    
class DiceHausdorffLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        dice = 1 - F.f_score(y_pr, y_gt, beta=self.beta, eps=self.eps, threshold=None, ignore_channels=self.ignore_channels)
        d_max = 0.03*max(y_pr.shape)
        hausdorff = torch.tensor(hausdorff_distance(y_pr.numpy(), y_gt.numpy()))/d_max
        print('dice =',dice.item(),' - hausdorff =', hausdorff.item())
        sum_loss = dice + hausdorff
        return sum_loss
