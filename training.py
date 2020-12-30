#!/usr/bin/env python
# coding: utf-8

# Packages

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import  utils
import random
import copy
import numpy as np
from models.bisenet import BiSeNet
from models.pranet import PraNet
from dataset import Dataset_SMP, get_preprocessing, get_training_augmentation, split_data
from display import display_result, display_proba
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.losses import DiceLoss, BCELoss
from loss import BCEDiceLoss, OhemBCELoss, SoftmaxFocalLoss, BinaryFocalLoss, SurfaceLoss
import re
import os

# Fix seed:

torch.backends.cudnn.deterministic = True
random.seed(123456)
torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)


# # Parameters

# In[5]:


# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
data_root = "C:\\Users\\gueganj\\Desktop\\My_DataBase\\nature\\"
# Models
model_name = "pranet"
# encoder
encoder = "mobilenet_v2"
# Number of classes in the dataset
num_classes = 1
# Batch size for training (change depending on how much memory you have)
batch_size = 8
# Learning rate
lr = 0.02
# Momentum
momentum = 0.99
# Weight decay
weight_decay = 0.0001
# algorithm use for optimization
algo_optim = 'SGD'
# Number of epochs to train for 
num_epochs = 200
# prediction threshold
threshold = 0.25
# size of image in input
input_size = [448,448]
# total number of image used
size_dataset = 136
# Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
feature_extract = False
# Flag for using Tensorboard tool
use_tensorboard = False
# Flag for using data augmentation
use_augmentation = True
# Flag for using a learning rate scheduler
lr_scheduler = "constant"
# Load checkpoint
load_checkpoint = False #"C:\\Users\\gueganj\\Desktop\\Eyeglasses Detection\\checkpoint_logs\\27_10_2020-11_50.ckpt"
# Landmarks directory
landmarks_dir = os.path.join(data_root,"landmarks","face_landmarks.csv")



# create a config dictionnary
config = {}
for item in dir().copy():
    if (not item.startswith('_')) and (item!='In') and (item!='Out') and item != 'config' and item != 'item':
        if str(type(eval(item)))[8:-2] in ['str','bool','int','float']:
            config[item] = eval(item)


# In[ ]:


if num_classes == 1:
    activation = 'sigmoid'
else:
    activation = 'softmax'

if not landmarks_dir:
    in_channels = 3
else :
    in_channels = 4
    
# # Device

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Model

# In[ ]:


# Initialize and Reshape the Networks
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if model_name == "bisenet":
    # Load
    model = BiSeNet(n_classes=19, activation="sigmoid") # trained on 19 classes
    file_path  = 'C:\\Users\\gueganj\\Desktop\\Eyeglasses Detection\\checkpoint_logs\\bisenet_CELEBAMASK.ckpt'
    model.load_state_dict(torch.load(file_path, map_location=device))
    # change final layer to tune and output only 2 classes
    model.conv_out.conv_out   = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)
    if landmarks_dir:
        model.cp.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

elif model_name == "unet":
    model = smp.Unet(encoder_name=encoder, encoder_weights='imagenet', activation=activation, classes=1, in_channels=in_channels)

elif model_name == "DeepLabV3Plus":
    model = smp.DeepLabV3Plus(encoder_name=encoder,  encoder_weights='imagenet', activation=activation)
    model.segmentation_head[0] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)

elif model_name == "pranet":
    model = PraNet()
    model.ra2_conv4.conv = nn.Conv2d(64, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.ra2_conv4.bn   = nn.BatchNorm2d(num_classes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
model.to(device)


# Data

# data split
image_extension = '.jpg'
mask_extension  = '.png'
train_set, valid_set, test_set = split_data(data_root, "images", image_extension, "masks\\frame", mask_extension, size_dataset, use_id=True)
train_image, train_mask = train_set
valid_image, valid_mask = valid_set
test_image, test_mask   = test_set
# create DataLoader
if use_augmentation:
    train_augmentation = get_training_augmentation()
else:
    train_augmentation = None
train_dataset = Dataset_SMP(train_image, train_mask, input_size, train_augmentation, get_preprocessing(preprocessing_fn), landmarks_dir)
valid_dataset = Dataset_SMP(valid_image, valid_mask, input_size, None, get_preprocessing(preprocessing_fn), landmarks_dir)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# In[ ]:


def denormalize(image, preprocessing_fn):
    image = image * torch.tensor(preprocessing_fn.keywords['std']).view(-1,1,1) + torch.tensor(preprocessing_fn.keywords['mean']).view(-1,1,1)
    return image


# # Optimizer

# In[ ]:

# Gather the parameters to be optimized/updated in this run : finetuning or feature extract
n_param_total = sum(p.numel() for p in model.parameters())
if feature_extract:
    print("Parameters to learn : ")
    params_to_update = []
    params_to_finetuned = re.compile('(segmentation_head.*)|(decoder.blocks.[4].*.)')
    for name, param in model.named_parameters():
        if params_to_finetuned.match(name):
            print(3*"\t", " - ", name)
            params_to_update.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    n_param_extract = sum(p.numel() for p in params_to_update)
    print('{:,}/{:,} = {:.2f} % parameters to learn'.format(n_param_extract,n_param_total,100*(n_param_extract/n_param_total)))  
else:
    params_to_update = model.parameters()
    print('\n ==> all {:,} parameters to learn'.format(n_param_total))  
          
# Optimisation method
if algo_optim == 'SGD':
    optimizer_ft = optim.SGD(params_to_update, lr, momentum)
elif algo_optim == 'Adam':
    optimizer_ft = optim.Adam(params_to_update, lr=lr)
elif algo_optim == 'RMSprop':
    optimizer_ft = optim.RMSprop(params_to_update, lr=lr)
elif algo_optim == 'ASGD':
    optimizer_ft = optim.ASGD(params_to_update, lr=lr)
elif algo_optim == 'Adamax':
    optimizer_ft = optim.Adamax(params_to_update, lr=lr)
elif algo_optim == 'Adagrad':
    optimizer_ft = optim.Adagrad(params_to_update, lr=lr)
elif algo_optim == 'Adadelta':
    optimizer_ft = optim.Adadelta(params_to_update, lr=lr)
# LR scheduler                
if lr_scheduler=='cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, len(train_loader))
elif lr_scheduler=='exponential':
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=0.5)
elif lr_scheduler=='reduceOnPlateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft)
elif lr_scheduler=='constant':
    scheduler = None

# # Tensorboard

# In[ ]:


date_time = datetime.now().strftime("%d_%m_%Y-%H_%M")
if use_tensorboard:
    writer    = SummaryWriter('tensorboard_logs/' + date_time)
    # to do : configure max_queue to SummaryWriter()
    images, labels = iter(train_loader).next()
    if landmarks_dir:
        landmarks = images[:,3,:,:].unsqueeze(1)
        images    = images[:,:3,:,:]
        lnd_grid  = utils.make_grid(landmarks, nrow=4, padding=10)
        writer.add_image('Landmarks batch', lnd_grid)
    if len(labels.shape)==4 and labels.shape[1]==4:
        labels = np.abs(1-labels[:,0,:,:].unsqueeze(1))
    if len(labels.shape)==4 and labels.shape[1]==3:
        labels = torch.sum(labels,dim=1).unsqueeze(1)
    images   = denormalize(images, preprocessing_fn)
    img_grid = utils.make_grid(images, nrow=4, padding=10, scale_each=True)
    lbl_grid = utils.make_grid(labels, nrow=4, padding=10)
    writer.add_image('Images batch', img_grid)
    writer.add_image('Labels batch', lbl_grid)
    writer.close

# # Loss

# BCEDiceLoss, OhemCELoss, SoftmaxFocalLoss, BinaryFocalLoss, SurfaceLoss, GeneralizedDice
#weight  =  20 * torch.ones((batch_size, num_classes, labels.shape[1], labels.shape[2]))
loss    = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=threshold)]


# # Checkpoint

# In[ ]:


if load_checkpoint:
    checkpoint = torch.load(load_checkpoint, map_location=device)
    # my own checkpoint
    if 'model' in checkpoint.keys():
        model.load_state_dict(checkpoint['model'])
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        epoch      = checkpoint['epoch']
        loss_value = checkpoint['loss']
        iou_score  = checkpoint['iou_score']
    # open source checkpoint
    else: 
        model.load_state_dict(checkpoint)
    print("checkpoint loaded !")


# # Training

# In[ ]:


# create epoch runners, it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer_ft, device=device, verbose=True)
valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=device, verbose=True)


# In[ ]:

inputs_0 = iter(valid_loader).next()[0][0,:,:,:].unsqueeze(0)


# In[ ]:


# train model for 40 epochs
best_score = 0
for epoch in range(1, num_epochs):
    
    print('\n {} Epoch {}/{} {}'.format('=' * 20, epoch, num_epochs, '=' * 20))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    if use_tensorboard:
        writer.add_scalar('loss/val', valid_logs[loss.__name__], epoch)
        writer.add_scalar('score/val', valid_logs['iou_score'], epoch)
        writer.add_scalar('loss/train', train_logs[loss.__name__], epoch)
        writer.add_scalar('score/train', train_logs['iou_score'], epoch)
        # do a prediction to display
        with torch.no_grad():
            prediction = model.forward(inputs_0)
            if landmarks_dir:
                inputs_0_ = inputs_0[:,:3,:,:]
            else:
                inputs_0_ = inputs_0[:,:3,:,:]
            images = denormalize(inputs_0_.squeeze(0), preprocessing_fn)
            writer.add_figure('Result', display_proba(images.permute(1,2,0), prediction), global_step=epoch)

    # save model
    if best_score < valid_logs['iou_score']:
        best_score = valid_logs['iou_score']
        best_model = copy.deepcopy(model)
        ckpt_path  = 'checkpoint_logs/'+date_time+'.ckpt'
        torch.save({'config':config,'epoch':epoch,'model':best_model.state_dict(),'optimizer':optimizer_ft.state_dict(),'loss':valid_logs[loss.__name__],'iou_score':valid_logs['iou_score']}, ckpt_path)
        if use_tensorboard:
            writer.add_hparams(config, {'hparam/IoU score':best_score})
        print('Model saved!')


# # Test best saved model

# In[ ]:

# create DataLoader
test_dataset = Dataset_SMP(test_image, test_mask, input_size, train_augmentation, get_preprocessing(preprocessing_fn), landmarks_dir)
test_loader  = torch.utils.data.DataLoader(test_dataset)


# In[ ]:


# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(model=best_model, loss=loss, metrics=metrics, device=device)
logs       = test_epoch.run(test_loader)

