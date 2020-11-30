#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import glob
from dataset import Dataset_SMP, get_preprocessing, get_training_augmentation, split_data
from display import display_result
import segmentation_models_pytorch as smp
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.bisenet import BiSeNet


# In[3]:


torch.backends.cudnn.deterministic = True
random.seed(123456)
torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)


# In[4]:


def denormalize(image, preprocessing_fn):
    image = image * torch.tensor(preprocessing_fn.keywords['std']).view(-1,1,1) + torch.tensor(preprocessing_fn.keywords['mean']).view(-1,1,1)
    return image


# In[5]:


use_augmentation = True
size_dataset = 118
landmarks_dir = None
batch_size = 2


# # Model

# In[27]:


input_size = (448,448)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[28]:


model = smp.Unet(encoder_name="mobilenet_v2",  encoder_weights='imagenet', activation='sigmoid') # Activation=None because I apply activation layer myself
model.segmentation_head[0] = nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.to(device)
preprocessing_fn = smp.encoders.get_preprocessing_fn("mobilenet_v2", 'imagenet')


# # Checkpoint

# In[29]:


checkpoint = torch.load('C:\\Users\\gueganj\\Desktop\\Eyeglasses Detection\\checkpoint_logs\\01_10_2020-16_27.ckpt')
model.load_state_dict(checkpoint['model'])
#model = torch.load('C:\\Users\\gueganj\\Desktop\\Eyeglasses Detection\\prediction\\model_frame.ckpt')
print("model loaded")


# # Data

# In[50]:


data_root = "C:\\Users\\gueganj\\Desktop\\My_DataBase\\nature\\face_cropped\\"
image_extension = '.jpg'
mask_extension  = '.png'
train_set, valid_set, test_set = split_data(data_root, "images", image_extension, "masks", mask_extension, size_dataset, use_id=True)
train_image, train_mask = train_set
valid_image, valid_mask = valid_set
test_image, test_mask   = test_set
# create DataLoader
if use_augmentation:
    train_augmentation = get_training_augmentation()
else:
    train_augmentation = None


# In[51]:


train_dataset = Dataset_SMP(train_image, train_mask, input_size, train_augmentation, get_preprocessing(preprocessing_fn), landmarks_dir)
valid_dataset = Dataset_SMP(valid_image, valid_mask, input_size, None, get_preprocessing(preprocessing_fn), landmarks_dir)
test_dataset  = Dataset_SMP(test_image, test_mask, input_size, None, get_preprocessing(preprocessing_fn), landmarks_dir)


# In[52]:


train_loader  = torch.utils.data.DataLoader(train_dataset)
valid_loader  = torch.utils.data.DataLoader(valid_dataset)
test_loader   = torch.utils.data.DataLoader(test_dataset)


# In[53]:


len(train_image),len(valid_image),len(test_image)


# # Evaluate

# In[54]:


threshold  = 0.1 
metrics    = [smp.utils.metrics.IoU(threshold=threshold)]
metric     = smp.utils.metrics.IoU(threshold=threshold)
loss       = smp.utils.losses.DiceLoss()


# ## on same size as training

# In[55]:



test_epoch = smp.utils.train.ValidEpoch(model=model, loss=loss, metrics=metrics, device=device)
logs       = test_epoch.run(test_loader)


# In[36]:


valid_epoch = smp.utils.train.ValidEpoch(model=model, loss=loss, metrics=metrics, device=device)
logs       = test_epoch.run(valid_loader)


# ## resize on original size after prediciton 

# In[17]:


from dataset import resize, get_tensor
from PIL import Image
import time


# In[18]:


score = []
for i, (image, mask) in enumerate(test_loader):
    with torch.no_grad():
        # prediction
        proba  = model.forward(image)
        # get original mask (not resized)
        original_mask = np.array(Image.open(test_dataset.masks_dir[i]).convert('P'))
        original_size = original_mask.shape[0:2]
        # resize prediction
        image_np      = image.squeeze().permute(1,2,0).numpy()
        proba_np      = proba.squeeze().numpy()
        sample        = resize(original_size, deform='rectangular')(image=image_np, mask=proba_np)
        new_proba_np  = sample['mask']
        # get score
        new_proba     = torch.from_numpy(new_proba_np)
        original_mask = torch.from_numpy(original_mask)
        score.append(metric(new_proba, original_mask))        


# In[19]:


print('score :',sum(score).numpy()/len(test_loader))


# ## resize on original size before prediction
# For segmentation, different size than training can be used but different result as to be expect

# ```score = []
# for i, (image, mask) in enumerate(test_loader):
#     with torch.no_grad():
#         # get original mask (not resized)
#         original_mask = np.array(Image.open(test_dataset.masks_dir[i]).convert('P'))
#         original_size = original_mask.shape[0:2]
#         # resize image
#         image_np   = image.squeeze().permute(1,2,0).numpy()
#         sample     = resize([500,500], deform='rectangular')(image=image_np)
#         new_image  = torch.from_numpy(sample['image']).permute(2,1,0).unsqueeze(0)
#         # prediction
#         proba  = model.forward(new_image)
#         print('OK !')
#         # get score
#         original_mask = torch.from_numpy(original_mask)
#         print(proba.shape,original_mask.shape)
#         score.append(metric(proba, original_mask))
# ```

# In[ ]:





# In[20]:


print('score :',sum(score).numpy()/len(test_loader))


# # Bad Prediction

# # Train

# In[37]:


x, prob, msk, score = [], [], [], []
for image, mask in tqdm(train_loader):
    with torch.no_grad():
        proba = model.forward(image)
        prob.append(proba)
        score.append(metric(proba, mask))
        x.append(image)
        msk.append(mask)


# In[38]:


index = np.argsort(score)


# In[43]:


for i in index[-10:]:
    img = denormalize(x[i].squeeze(0), preprocessing_fn)
    display_result(img.permute(1,2,0).numpy(), msk[i].squeeze(), prob[i].squeeze(), threshold, metric, display=True)


# # Test

# In[40]:


x, prob, msk, score = [], [], [], []
for image, mask in tqdm(test_loader):
    with torch.no_grad():
        proba = model.forward(image)
        prob.append(proba)
        score.append(metric(proba, mask))
        x.append(image)
        msk.append(mask)


# In[41]:


index = np.argsort(score)


# In[42]:


for i in index[:10]:
    img = denormalize(x[i].squeeze(0), preprocessing_fn)
    display_result(img.permute(1,2,0).numpy(), msk[i].squeeze(), prob[i].squeeze(), threshold, metric, display=True)


# In[ ]:





# In[ ]:




