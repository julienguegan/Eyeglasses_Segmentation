from torchvision import utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def display_batch(batch_tensor, nrow=5):
    # if label add a dimension and *255 to be visible
    if len(batch_tensor.shape)==3:
        batch_tensor = 255*batch_tensor.unsqueeze(1)
    # make grid (2 rows and 5 columns) to display our 10 images
    if batch_tensor.shape[1]!=3:
         batch_tensor = batch_tensor.permute(0,3,1,2)
    grid_img = utils.make_grid(batch_tensor, nrow=nrow, padding=10, scale_each=True)
    # reshape and plot (because plt needs channel as the last dimension)
    plt.figure(figsize=(15,5))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(grid_img.shape)
    plt.axis('off')
    plt.show()
    
def display_segmentation(image, label, colorbar=False):
    fig = plt.figure(figsize=(12, 10))
    mask = np.ma.masked_where(label == 0, label)
    plt.imshow(image)
    plt.imshow(mask, cmap='jet', alpha=0.5) # interpolation='none'
    plt.axis('off')
    if colorbar: # for logit
        plt.colorbar()
    return fig
   
def display_proba(image, proba):
    if image.max() < 100:
        vmax = 1
    mask_proba = np.ma.masked_where(proba < 0.001, proba)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(image, vmin=0, vmax=vmax)
    plt.title('probabilities')
    plt.imshow(mask_proba, cmap='jet', alpha=0.5) # interpolation='none'
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    return fig

def display_result(image, true_label, proba, threshold, metric, display=False):
    
    if image.max() < 100:
        vmax = 1
    
    fig = plt.figure(figsize=(15, 10))
    # ground truth
    plt.subplot(131)
    plt.imshow((255*image).astype('uint8'), vmin=0, vmax=vmax) # interpolation='none'
    plt.axis('off')
    plt.title('Image')
    # probabilities
    mask_proba = proba#np.ma.masked_where(proba < 0.001, proba)
    plt.subplot(132)
    plt.imshow(image.astype('uint8'), vmin=0, vmax=vmax)
    plt.title('probabilities')
    plt.imshow(mask_proba, cmap='jet', alpha=0.5) # interpolation='none'
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    # prediction
    plt.subplot(133)
    probas = proba.squeeze().detach().cpu().numpy()
    label  = true_label.squeeze().detach().cpu().numpy()
    prediction      = (probas >= threshold)
    good_prediction = (label == prediction)
    bad_prediction  = (label != prediction)
    TP = good_prediction & (label == 1.)
    TN = good_prediction & (label == 0.)
    FP = bad_prediction & (label == 1.)
    FN = bad_prediction & (label == 0.)
    mask_predict = np.zeros((label.shape[0],label.shape[1],3)).astype(np.uint8)
    colors = np.array([[0,255,0], [0,0,255], [255,0,0], [255,140,0]])
    mask_predict[TP] = colors[0] # green
    mask_predict[TN] = colors[1] # blue
    mask_predict[FP] = colors[2] # red
    mask_predict[FN] = colors[3] # orange
    n = label.size
    label_type = ["TP : {:2.1f} %".format(100*TP.sum()/n),"TN : {:2.1f} %".format(100*TN.sum()/n),"FP : {:2.1f} %".format(100*TP.sum()/n),"FN : {:2.1f} %".format(100*FN.sum()/n)]
    legend     = [patches.Patch(color=colors[i]/255, label="{}".format(label_type[i])) for i in range(len(colors))]
    score      = metric(proba, true_label)
    plt.imshow(mask_predict, interpolation='none')
    plt.title('IoU score : {:.2f} %'.format(100*score.numpy()))
    plt.legend(handles=legend, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.xticks(()),plt.yticks(())
    if display:
        plt.show()
    
    return fig