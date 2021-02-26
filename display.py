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
    
def display_segmentation(image, label, score=None):
    fig  = plt.figure(figsize=(12, 10))
    mask = np.ma.masked_where(label == 0, label)
    image_mean = np.mean(image,axis=2)
    image_mean = np.repeat(image_mean[:, :, np.newaxis], 3, axis=2)
    image_mean[mask.squeeze()] = [1,0,0]
    plt.imshow(image_mean, cmap='gray',vmin=0,vmax=1)
    plt.axis('off')
    if score:
        plt.title(score)
    return fig
   
def display_proba(image, proba):
    if image.max() < 100:
        vmax = 1
    mask_proba = np.ma.masked_where(proba < 0.001, proba)
    n = proba.shape[1]
    fig = plt.figure(figsize=(8*n,6))
    image_mean = np.mean(image.permute(1,2,0).numpy(),axis=2)
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(image_mean, vmin=0, vmax=vmax)
        plt.imshow(mask_proba[0,i,:,:].squeeze(), cmap='jet', alpha=0.3) # interpolation='none'
        plt.axis('off')
        if n > 1:
            plt.title("class " + str(i))
            plt.suptitle("probabilities",fontsize=25)
        else:
            plt.title('probabilities')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
    return fig

def display_result(image, true_label, proba, threshold, metric, display=False, legend=False):
    
    if image.max() < 100:
        vmax = 1
    
    fig = plt.figure(figsize=(15, 5))
    # ground truth
    plt.subplot(121)
    #plt.imshow((255*image).astype('uint8'), vmin=0, vmax=vmax) # interpolation='none'
    probas = proba.squeeze()
    label  = true_label.squeeze().numpy()
    prediction      = (probas >= threshold).numpy().astype(float)
    good_prediction = (label == prediction)
    bad_prediction  = (label != prediction)
    TP = good_prediction & (label == 1.)
    TN = good_prediction & (label == 0.)
    FN = bad_prediction & (label == 1.)
    FP = bad_prediction & (label == 0.)
    colors = np.array([[0,255,0], [0,0,255], [255,0,0], [255,140,0]])/255
    mask_predict = image.copy()
    mask_predict[TP,:] = colors[0] # green
    mask_predict[FP,:] = colors[2] # red
    mask_predict[FN,:] = colors[3] # orange
    label_type = ["TP","TN","FP","FN"]
    legend     = [patches.Patch(color=colors[i], label="{}".format(label_type[i])) for i in range(len(colors))]
    score      = metric(proba, true_label)
    plt.imshow(mask_predict)
    plt.title('IoU score : {:.2f} %'.format(100*score.numpy()))
    if legend:
        plt.legend(handles=legend, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.xticks(()),plt.yticks(())
    plt.axis('off')
    # probabilities
    mask_proba = proba.numpy()#np.ma.masked_where(proba < 0.001, proba)
    plt.subplot(122)
    plt.title('probabilities')
    plt.imshow(image)
    plt.imshow(mask_proba, cmap='jet', alpha=0.5) # interpolation='none'
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    if display:
        plt.show()
    
    return fig