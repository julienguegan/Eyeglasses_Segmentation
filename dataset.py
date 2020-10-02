import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import utils
import matplotlib.pyplot as plt
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

class CustomDataset(Dataset):
    
    def __init__(self, image_paths, target_paths, transform=None):
        self.image_paths  = image_paths
        self.target_paths = target_paths
        self.to_tensor    = transforms.ToTensor()
        self.transform    = transform
    
    def __getitem__(self, index):
        # open as image
        image = Image.open(self.image_paths[index])
        mask  = Image.open(self.target_paths[index])
        # augmentation
        if self.transform is not None:
            seed = np.random.randint(123456789)
            random.seed(seed), torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed), torch.manual_seed(seed)
            mask  = self.transform(mask) 
        # transform image to tensor
        #image = self.to_tensor(image) 
        # format mask to class
        mask = np.array(mask)[0,:,:]
        mask = np.where(mask>mask.max()*0.4, 1, 0)
        mask = torch.from_numpy(mask) 
        mask = mask.float()
        return image, mask

    def __len__(self):
        return len(self.image_paths)


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
    
def display_segmentation(image, label, colorbar=False):
    fig = plt.figure(figsize=(12, 10))
    mask = np.ma.masked_where(label == 0, label)
    plt.imshow(image)
    plt.imshow(mask, cmap='jet', alpha=0.5) # interpolation='none'
    plt.axis('off')
    if colorbar: # for logit
        plt.colorbar()
    return fig
    

def display_result(image, label, predict, proba):
    
    fig = plt.figure(figsize=(15, 10))
    mask_predict = np.ma.masked_where(predict == 0, predict)
    # ground truth
    plt.subplot(131)
    plt.imshow(label) # interpolation='none'
    plt.axis('off')
    plt.title('ground truth')
    # prediction
    plt.subplot(132)
    plt.imshow(image)
    plt.imshow(mask_predict, cmap='jet', alpha=0.5) # interpolation='none'
    plt.axis('off')
    plt.title('prediction')
    # probabilities
    plt.subplot(133)
    plt.imshow(image)
    plt.title('probabilities')
    plt.imshow(proba, cmap='jet', alpha=0.5) # interpolation='none'
    plt.axis('off')
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)
    
    return fig
