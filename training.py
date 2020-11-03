#!/usr/bin/env python
# coding: utf-8

# # Packages


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import  utils
import random
import os
import copy
import glob
import numpy as np
from bisenet import BiSeNet
from dataset import Dataset_SMP, get_preprocessing, get_validation_augmentation, get_training_augmentation, split_data
from display import display_result, display_proba
from loss import OhemCELoss, BCELoss2d, DiceLoss, CrossEntropyLoss2d, NLLLoss2d
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp


# In[4]:


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
model_name = "bisenet"
# encoder
encoder = "mobilenet_v2"
# Number of classes in the dataset
num_classes = 1
# Batch size for training (change depending on how much memory you have)
batch_size = 8
# Learning rate
lr = 0.001
# Momentum
momentum = 0.99
# Weight decay
weight_decay = 0.0001
# algorithm use for optimization
algo_optim = 'SGD'
# Number of epochs to train for 
num_epochs = 200
# prediction threshold
threshold = 0.5
# size of image in input
input_size = 448
# total number of image used
size_dataset = 36
# Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
feature_extract = False
# Flag for using Tensorboard tool
use_tensorboard = True
# Flag for using data augmentation
use_augmentation = True
# Flag for using a learning rate scheduler
lr_scheduler = "constant"
# Load checkpoint
load_checkpoint = "C:\\Users\\gueganj\\Desktop\\Eyeglasses Detection\\checkpoint_logs\\02_11_2020-14_30.ckpt"
# Landmarks directory
landmarks_dir = os.path.join(data_root,"landmarks","face_landmarks.csv")


# In[6]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Model

# In[7]:


# Initialize and Reshape the Networks
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if model_name == "bisenet":
    # Load
    file_path  = 'C:\\Users\\gueganj\\Desktop\\Face_Parsing\\face parsing - PyTorch\\res\\cp\\79999_iter.pth'
    model = BiSeNet(n_classes=19) # trained on 19 classes
    model.load_state_dict(torch.load(file_path, map_location=device))
    print("model loaded !")
    # change final layer to tune and output only 2 classes
    set_parameter_requires_grad(model, feature_extract)
    model.conv_out.conv_out   = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #model.conv_out16.conv_out = nn.Conv2d(64, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #model.conv_out32.conv_out = nn.Conv2d(64, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)
    if landmarks_dir:
        model.cp.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    for name, param in model.ffm.named_parameters():
        param.requires_grad = True
    for name, param in model.cp.conv_head16.named_parameters():
        param.requires_grad = True
    for name, param in model.cp.conv_head32.named_parameters():
        param.requires_grad = True
    for name, param in model.cp.resnet.layer4[1].named_parameters():
        param.requires_grad = True
elif model_name == "unet":
    model = smp.Unet(encoder_name=encoder,  encoder_weights='imagenet', activation='sigmoid') # Activation=None because I apply activation layer myself
    model.segmentation_head[0] = nn.Conv2d(16, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False)
    if landmarks_dir:
        model.encoder.features[0][0] = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
model.to(device)

print()


# # Data

# In[8]:


# data split
image_extension = '.jpg'
mask_extension  = '.png'
train_set, valid_set, test_set = split_data(data_root, "images", image_extension, "masks\\frame", mask_extension, size_dataset, use_id=False)
train_image = train_set[0]
train_mask  = train_set[1]
valid_image = valid_set[0]
valid_mask  = valid_set[1]
test_image  = test_set[0]
test_mask   = test_set[1]
# create DataLoader
if use_augmentation:
    train_augmentation = get_training_augmentation(input_size,input_size)
else:
    train_augmentation = get_validation_augmentation(input_size,input_size)
train_dataset = Dataset_SMP(train_image, train_mask, augmentation=train_augmentation, preprocessing=get_preprocessing(preprocessing_fn), landmarks_dir=landmarks_dir)
valid_dataset = Dataset_SMP(valid_image, valid_mask, augmentation=get_validation_augmentation(input_size,input_size), preprocessing=get_preprocessing(preprocessing_fn), landmarks_dir=landmarks_dir)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# In[9]:


def denormalize(image, preprocessing_fn):
    image = image * torch.tensor(preprocessing_fn.keywords['std']).view(-1,1,1) + torch.tensor(preprocessing_fn.keywords['mean']).view(-1,1,1)
    return image


# # Optimizer

# In[10]:


# Gather the parameters to be optimized/updated in this run : finetuning or feature extract
params_to_update = model.parameters()
print("Parameters to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

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
                  
if lr_scheduler=='cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, len(train_loader))
elif lr_scheduler=='exponential':
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer_ft, gamma=1.5)
elif lr_scheduler=='reduceOnPlateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft)
elif lr_scheduler=='constant':
    scheduler = None


# # Loss

# In[11]:


loss      = smp.utils.losses.DiceLoss()
metrics   = [smp.utils.metrics.IoU(threshold=threshold)]


# # Tensorboard

# In[12]:


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

    images   = denormalize(images, preprocessing_fn)
    img_grid = utils.make_grid(images, nrow=4, padding=10, scale_each=True)
    lbl_grid = utils.make_grid(labels, nrow=4, padding=10)
    writer.add_image('Images batch', img_grid)
    writer.add_image('Labels batch', lbl_grid)
    writer.close


# # Checkpoint

# In[13]:


if load_checkpoint:
    checkpoint = torch.load(load_checkpoint)
    model.load_state_dict(checkpoint['model'])
    optimizer_ft.load_state_dict(checkpoint['optimizer'])
    epoch      = checkpoint['epoch']
    loss_value = checkpoint['loss']
    iou_score  = checkpoint['iou_score']


# # Training

# In[14]:


# create epoch runners, it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer_ft, device=device, verbose=True)
valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, device=device, verbose=True)


# In[15]:


inputs_0 = iter(valid_loader).next()[0][0,:,:,:].unsqueeze(0)


# In[ ]:


# train model for 40 epochs
best_score = 0
for epoch in range(1, num_epochs):
    
    print('\n {} Epoch {}/{} {}'.format('=' * 20, epoch, num_epochs, '=' * 20))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    if use_tensorboard:
        writer.add_scalar('loss/val', valid_logs['dice_loss'], epoch)
        writer.add_scalar('score/val', valid_logs['iou_score'], epoch)
        writer.add_scalar('loss/train', train_logs['dice_loss'], epoch)
        writer.add_scalar('score/train', train_logs['iou_score'], epoch)
        # do a prediction to display
        with torch.no_grad():
            prediction = model.forward(inputs_0)
            if landmarks_dir:
                inputs_0_ = inputs_0[:,:3,:,:]
            else:
                inputs_0_ = inputs_0[:,:3,:,:]
            images = denormalize(inputs_0_.squeeze(0), preprocessing_fn)
            writer.add_figure('Result', display_proba(images.permute(1,2,0), prediction.squeeze()), global_step=epoch)

    # save model
    if best_score < valid_logs['iou_score']:
        best_score = valid_logs['iou_score']
        best_model = copy.deepcopy(model)
        ckpt_path  = 'checkpoint_logs/'+date_time+'.ckpt'
        torch.save({'epoch':epoch,'model':best_model.state_dict(),'optimizer':optimizer_ft.state_dict(),'loss':valid_logs['dice_loss'],'iou_score':valid_logs['iou_score']}, ckpt_path)
        print('Model saved!')


# In[ ]:


writer.add_hparams({"Image size":int(input_size),
                   "data":os.path.basename(os.path.normpath(data_root)),
                   "Dataset size":int(size_dataset),
                   "Architecture":model_name,
                   "Learning rate":lr,
                   "weight_decay":weight_decay,
                   "Momentum":momentum,
                   "LR scheduler": lr_scheduler.__class__.__name__ if lr_scheduler != 'constant' else 'constant',
                   "Optimisation algorithm":optimizer_ft.__class__.__name__,
                   "Epoch number":int(num_epochs),
                   "Batch size":int(batch_size),
                   "Loss":loss.__class__.__name__,
                   "Threshold sigmoid":threshold}, 
                   {'hparam/IoU score':best_score})


# # Test best saved model

# In[ ]:

# create DataLoader
test_dataset = Dataset_SMP(test_image, test_mask, augmentation=get_validation_augmentation(input_size,input_size), preprocessing=get_preprocessing(preprocessing_fn))
test_loader  = torch.utils.data.DataLoader(test_dataset)


# In[ ]:


# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(model=best_model, loss=loss, metrics=metrics, device=device)
logs       = test_epoch.run(test_loader)


# # Display Prediction

# ## train

# In[ ]:


train_dataset = Dataset_SMP(train_image[:10], train_mask[:10], augmentation=train_augmentation, preprocessing=get_preprocessing(preprocessing_fn))
train_loader  = torch.utils.data.DataLoader(train_dataset)


# In[ ]:


for x,y in train_loader:
   with torch.no_grad():
       proba = best_model.forward(x)
   image = denormalize(x.squeeze(0), preprocessing_fn)
   # display
   display_result(image.permute(1,2,0).numpy(), y, proba.squeeze(), threshold, metrics[0], display=True)


# ## valid

# In[ ]:


valid_dataset = Dataset_SMP(valid_image[:10], valid_mask[:10], augmentation=train_augmentation, preprocessing=get_preprocessing(preprocessing_fn))
valid_loader  = torch.utils.data.DataLoader(valid_dataset)


# In[ ]:


for x,y in test_loader:
   with torch.no_grad():
       proba = best_model.forward(x)
   image = denormalize(x.squeeze(0), preprocessing_fn)
   # display
   display_result(image.permute(1,2,0).numpy(), y, proba.squeeze(), threshold, metrics[0], display=True)


# ## test

# In[ ]:


test_loader  = torch.utils.data.DataLoader(test_dataset)


# In[ ]:


for x,y in test_loader:
   with torch.no_grad():
       proba = best_model.forward(x)
   image = denormalize(x.squeeze(0), preprocessing_fn)
   # display
   display_result(image.permute(1,2,0).numpy(), y, proba.squeeze(), threshold, metrics[0], display=True)





