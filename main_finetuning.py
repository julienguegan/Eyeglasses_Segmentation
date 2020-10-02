#!/usr/bin/env python
# coding: utf-8

# ====================== Packages ====================== #

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import random
import time
import os
import copy
import glob
from bisenet import BiSeNet
from dataset import CustomDataset, display_batch, display_segmentation, display_result
from loss import OhemCELoss, BCELoss2d, DiceLoss, CrossEntropyLoss2d, NLLLoss2d
from metrics import IoU_score
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from colorama import init
init()

# ====================== Parameters ============================ #

# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
data_root = "C:\\Users\\gueganj\\Desktop\\My_DataBase\\shuang_data\\"
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "unet"
# Number of classes in the dataset
num_classes = 1
# Batch size for training (change depending on how much memory you have)
batch_size = 8
# Learning rate
lr = 0.05
# Momentum
momentum = 0.99
# Weight decay
wd = 0
# Number of epochs to train for 
num_epochs = 100
# prediction threshold
threshold = 0.25
# size of image in input
input_size = 224
# total number of image used
size_dataset = 100
# Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
feature_extract = False
# Flag for using Tensorboard tool
use_tensorboard = True
# Flag for using data augmentation
use_augmentation = False
# Flag for using a learning rate scheduler
use_scheduler = "cosine"
# Load checkpoint
load_checkpoint = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ======================  Model  ============================ #

# Initialize and Reshape the Networks
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if model_name == "bisenet":
    # Load
    file_path  = 'C:\\Users\\gueganj\\Desktop\\face parsing - PyTorch\\res\\cp\\79999_iter.pth'
    model = BiSeNet(n_classes=19) # trained on 19 classes
    model.load_state_dict(torch.load(file_path, map_location=device))
    # change final layer to tune and output only 2 classes
    set_parameter_requires_grad(model, feature_extract)
    model.conv_out.conv_out   = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.conv_out16.conv_out = nn.Conv2d(64, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)
    model.conv_out32.conv_out = nn.Conv2d(64, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)
elif model_name == "unet":
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name='mobilenet_v2', activation=None) # Activation=None because I apply activation layer myself
    model.segmentation_head[0] = nn.Conv2d(16, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)

model.to(device)

# ====================== Data ====================== #

# data augmentation
if use_augmentation:
    transform = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                        resample=2,
                        shear=5),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
                    #transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
else:
    transform = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor()])
    
# path
folder_data = glob.glob(os.path.join(data_root,"images\\*.png"))
folder_mask = glob.glob(os.path.join(data_root,"masks\\*.png"))
# suffle the 2 lists the same way (to be sure)
lists_shuffled = list(zip(folder_data, folder_mask))
random.shuffle(lists_shuffled)
folder_data, folder_mask = zip(*lists_shuffled)
# split in train/test
train_size   = 0.8
train_image_paths = folder_data[:int(size_dataset*train_size)]
test_image_paths  = folder_data[int(size_dataset*train_size):]
train_mask_paths  = folder_mask[:int(size_dataset*train_size)]
test_mask_paths   = folder_mask[int(size_dataset*train_size):]
# create DataLoader
train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform)
test_dataset  = CustomDataset(test_image_paths, test_mask_paths)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader   = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# ====================== Optimizer ====================== #

# Gather the parameters to be optimized/updated in this run : finetuning or feature extract
params_to_update = model.parameters()
print("Parameters to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(params_to_update, lr, momentum)
optimizer_ft = optim.Adam(params_to_update, lr=lr, weight_decay=wd)
#optimizer_ft = optim.RMSprop(params_to_update, lr=lr)
#optimizer_ft = optim.ASGD(params_to_update, lr=lr)
#optimizer_ft = optim.Adamax(params_to_update, lr=lr)
#optimizer_ft = optim.Adagrad(params_to_update, lr=lr)
#optimizer_ft = optim.Adadelta(params_to_update)
                  
if use_scheduler:
    if use_scheduler=='cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, len(train_loader))
    elif use_scheduler=='exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=1.5)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft)

# ====================== Loss ====================== #

# criterion = nn.CrossEntropyLoss() #BCELoss2d() #DiceLoss() #CrossEntropyLoss2d() #DiceLoss() #NLLLoss2d
criterion = nn.BCEWithLogitsLoss()

# ====================== Training ====================== #
date_time = datetime.now().strftime("%d_%m_%Y-%H_%M")
if use_tensorboard:
    writer    = SummaryWriter('tensorboard_logs/' + date_time)
    # to do : configure max_queue to SummaryWriter()
    images, labels = iter(train_loader).next()
    img_grid = utils.make_grid(images, nrow=4, padding=10)
    lbl_grid = utils.make_grid(labels.unsqueeze(1), nrow=4, padding=10)
    writer.add_image('Images batch', img_grid)
    writer.add_image('Labels batch', lbl_grid)
    writer.add_graph(model, images)
    writer.close

# ====================== Checkpoint ====================== #

if load_checkpoint:
    checkpoint = torch.load(load_checkpoint)
    model.load_state_dict(checkpoint['model'])
    optimizer_ft.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss  = checkpoint['loss']
    score = checkpoint['IoU_score']


def train_model(model, dataloaders, optimizer, criterion, num_epochs=25, threshold=0.5):
    since = time.time()

    val_acc_history = []
    
    best_model = copy.deepcopy(model.state_dict())
    best_acc   = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            # Iterate over data.
            running_loss, running_iou = 0.0, 0.0
            for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # FORWARD
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    output = model(inputs)
                    logits = output.squeeze(1).detach()
                    predictions = torch.sigmoid(logits) > threshold
                    loss = criterion(output.squeeze(1), labels)
                    
                    # BACKWARD (in training phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if use_scheduler:
                            scheduler.step()
                        
                # statistics
                running_loss += loss.item() #* inputs.size(0)
                running_iou  += IoU_score(predictions, labels)
                if use_tensorboard and (i%int(batch_size)==0):
                    writer.add_scalar('loss/'+phase, running_loss/(i+1), 1+i+epoch*len(dataloaders[phase]))
                    writer.add_scalar('score/'+phase, running_iou/(i+1), 1+i+epoch*len(dataloaders[phase]))
                    inputs_0  = inputs[0,:,:,:].permute(1,2,0)
                    predict_0 = predictions[0,:,:]
                    proba_0   = torch.sigmoid(logits[0,:,:])
                    label_0   = labels[0,:,:]
                    writer.add_figure('Result', display_result(inputs_0, label_0, predict_0, proba_0), global_step=1+i+epoch*len(dataloaders[phase]))
                    if use_scheduler:
                        writer.add_scalar('lr/'+phase, scheduler.get_last_lr()[0], 1+i+epoch*len(dataloaders[phase]))

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_iou  = running_iou  / len(dataloaders[phase])

            tqdm.write('{:5s} : Loss={:.4f} - IoU={:.4f}'.format(phase, epoch_loss, epoch_iou))

                
            # deep copy the model
            if phase == 'val' and epoch_iou > best_acc:
                best_acc   = epoch_iou
                best_model = copy.deepcopy(model)
                ckpt_path  = 'checkpoint_logs/'+date_time+'.ckpt'
                torch.save({'epoch':epoch,'model':best_model.state_dict(),'optimizer':optimizer.state_dict(),'loss':epoch_loss,'IoU_score':epoch_iou}, ckpt_path)
            if phase == 'val':
                val_acc_history.append(epoch_iou)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Accuracy: {:4f}'.format(best_acc))
    
    return best_model, best_acc


# In[17]:


# loaders dict
dataloaders_dict = {}
dataloaders_dict['train'] = train_loader
dataloaders_dict['val']   = test_loader

# Train and evaluate 
model_ft, best_acc = train_model(model, dataloaders_dict, optimizer_ft, criterion, num_epochs, threshold)


writer.add_hparams({"Image size":int(input_size),
                   "shuang/Julien":"shuang",
                   "Dataset size":int(size_dataset),
                   "Architecture":model_name,
                   "Learning rate":lr,
                   "Momentum":momentum,
                   "LR scheduler": scheduler.__class__.__name__ if use_scheduler else 'None',
                   "Optimisation algorithm":optimizer_ft.__class__.__name__,
                   "Epoch number":int(num_epochs),
                   "Batch size":int(batch_size),
                   "Loss":criterion.__class__.__name__,
                   "Threshold sigmoid":threshold}, 
                   {'hparam/IoU score':best_acc})

'''
score_thres = 0.7
ignore_idx  = -100
n_min = 16 * 448 * 448//16
LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
#lossp, loss2, loss3 = LossP(out, labels), Loss2(out16, labels), Loss3(out32, labels)
#loss                = lossp + loss2 + loss3
'''
