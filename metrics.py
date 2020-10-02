#!/usr/bin/env python
# coding: utf-8
from torchvision import utils
import matplotlib.pyplot as plt
import torch

def display_batch(batch_tensor, nrow=5):
    
    # if label add a dimension and *255 to be visible
    if len(batch_tensor.shape)==3:
        batch_tensor = 255*batch_tensor.unsqueeze(1)
    # make grid (2 rows and 5 columns) to display our 10 images
    grid_img = utils.make_grid(batch_tensor, nrow=nrow, padding=10)
    # reshape and plot (because MPL needs channel as the last dimension)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()



def IoU_score(predictions, labels):

    #predictions.requires_grad = False
    predictions = predictions.detach()
    intersection = torch.logical_and(predictions, labels).float().sum((1, 2))
    union        = torch.logical_or(predictions, labels).float().sum((1, 2))
    #intersection = (predictions & labels).float().sum((1, 2))
    #union        = (predictions | labels).float().sum((1, 2))
    iou = intersection / union
    
    return iou.mean()