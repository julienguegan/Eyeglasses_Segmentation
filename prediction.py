#!/usr/bin/env python
# coding: utf-8



import torch
import segmentation_models_pytorch as smp
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as albu
import argparse
from tqdm import tqdm

def denormalize(image, preprocessing_fn):
    """From image in [0,1] to [0,255] according to preprocessing function"""
    image = image * torch.tensor(preprocessing_fn.keywords['std']).view(-1,1,1) + torch.tensor(preprocessing_fn.keywords['mean']).view(-1,1,1)
    return image

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    input : 
        - preprocessing_fn : preprocessing function
    output :
        - transform : albumentations object
    """
    return albu.Lambda(image=preprocessing_fn)

def resize(input_size, deform="rectangular"):
    """Transformation for validation set
    input : 
        - input_size : integer for resizing maximum side
    output :
        - transform : albumentations object
    """
    if deform=="square":
        transform = albu.Resize(input_size, input_size)
    elif deform=="scale":
        transform = albu.LongestMaxSize(input_size)
    elif deform=="rectangular":
        transform = albu.Resize(input_size[0], input_size[1])
    else:
        print("deform argument unknown")
        
    return transform

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    
    # argument parser
    parser = argparse.ArgumentParser(description="Prediction from CNN model")
    parser.add_argument( "--input_dir", default="prediction\\some_images", help="path to images directory, all images inside are used.", type=str)
    parser.add_argument( "--output_dir", default="prediction", help="path to output directory where prediction will be saved in.", type=str)
    parser.add_argument( "--model_dir", default="prediction", help="path to model directory used for prediction.", type=str)
    parser.add_argument( "--zoom", default=True, help="True/False to zoom the output prediction to eyeglasses zone", type=bool)    
    args = parser.parse_args()
    
    # size to downsample image as in training
    input_size = (544,960)

    # Load Model
    model = torch.load(os.path.join(args.model_dir,'model_frame.ckpt'), map_location=device)
    # /!\ for model_edges.ckpt, threshold should be set at 0.15. Otherwise, 0.5 is good
    model.eval()
    model.to(device)

    # Normalize image
    preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet18", 'imagenet')

    list_image = glob.glob(os.path.join(args.input_dir,'*.jpg'))
    if not list_image:
        print("ERROR : no image found in input")
    for image_name in tqdm(list_image):
        # read
        image = np.array(Image.open(image_name))
        # resize
        sample = resize(input_size)(image=image)
        # apply preprocessing
        sample = get_preprocessing(preprocessing_fn)(image=sample['image'])
        # transform to tensor format
        image = torch.Tensor(sample['image'].transpose(2, 0, 1).astype('float32')).unsqueeze(0)
        # prediction
        with torch.no_grad():
            proba = model.forward(image)
            prediction = proba > 0.5
            prediction = prediction.squeeze()
        # format for segmentation display (currently only 1 class)
        image = denormalize(image.squeeze(), preprocessing_fn)
        image = image.permute(1,2,0).numpy()
        image_mean = np.mean(image,axis=2)
        image_mean = np.repeat(image_mean[:, :, np.newaxis], 3, axis=2)
        image_mean[prediction] = [1,0,0]
        image_mean[image_mean<0] = 0
        # cropped to eyeglasses zone if requires
        if args.zoom:
            idx = np.where(prediction)
            x_max, x_min = idx[0].max(), idx[0].min()
            y_max, y_min = idx[1].max(), idx[1].min()
            image_mean = image_mean[x_min-10:x_max+10,y_min-10:y_max+10]
        # save and display
        plt.imsave(os.path.join(args.output_dir,os.path.basename(image_name)),image_mean)
        #plt.imshow(image_mean, cmap='gray',vmin=0,vmax=1)
        #plt.axis('off')
        #plt.show()
        