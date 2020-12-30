import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
import random
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import pandas as pd
import os
import glob

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


class Dataset_SMP(BaseDataset):
    """Read images, apply augmentation and preprocessing transformations.    
    input : 
        - images_dir    (str)          : path to images folder
        - masks_dir     (str)          : path to segmentation masks folder
        - augmentation  (albu.Compose) : data transfromation pipeline (e.g. flip, scale, etc.)
        - preprocessing (albu.Compose) : data preprocessing (e.g. normalization, shape manipulation, etc.)
        - keypoints     (str)          : path to keypoints face
    
    """
    
    def __init__(self, images_dir, masks_dir, input_size=448, augmentation=None, preprocessing=None, landmarks_dir=None):
        self.images_dir    = images_dir
        self.masks_dir     = masks_dir
        self.preprocessing = preprocessing
        self.input_size    = input_size
        if landmarks_dir:
            self.use_landmarks  = True
            self.landmarks_file = pd.read_csv(landmarks_dir)
            if augmentation:
                self.augmentation = albu.Compose(augmentation, keypoint_params=albu.KeypointParams(format='xy'))
            else:
                self.augmentation = augmentation
        else:
            self.use_landmarks  = False
            self.augmentation   = augmentation
    
    def __getitem__(self, i):
        # open image 
        image = np.array(Image.open(self.images_dir[i]))
        mask  = np.array(Image.open(self.masks_dir[i]).convert('P'))
        if os.path.basename(self.images_dir[i]).replace('.jpg','').replace('.png','') != os.path.basename(self.masks_dir[i]).replace('.jpg','').replace('.png',''):
            print('ERROR : name mask != image')
        if image.shape[0:2] != mask.shape :
            print('ERROR : size mask != image')  
        
        # format mask to class
        if '.jpg' in os.path.basename(self.masks_dir[i]):
            mask = np.array(mask)
            mask = np.where(mask>mask.max()*0.4, 1., 0.)
        
        # get landmarks
        if self.use_landmarks:
            landmarks = self.landmarks_file[self.landmarks_file['image_name'] == os.path.basename(self.images_dir[i])].iloc[:,1:]
            landmarks = np.array(landmarks).reshape(-1, 2)

        # resize all
        if self.input_size:
            if self.use_landmarks:
                # do : albu compose
                sample = albu.Compose([resize(self.input_size)], keypoint_params=albu.KeypointParams(format='xy'))(image=image, mask=mask, keypoints=landmarks)
                image, mask, landmarks = sample['image'], sample['mask'], np.array(sample['keypoints'])
            else:
                sample = resize(self.input_size)(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']                        
         
        # apply augmentations
        if self.augmentation:
            color_mask = OnlyOnMask([albu.RGBShift([-150, 150], p=0.1)])
            sample     = color_mask(image=image, mask=mask)
            image      = sample['image']
            if self.use_landmarks:
                sample     = self.augmentation(image=image, mask=mask, keypoints=landmarks)
                image, mask, landmarks = sample['image'], sample['mask'], np.array(sample['keypoints'])
            else:
                sample      = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
                
        # apply preprocessing
        if self.preprocessing:
            sample      = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # stack heatmap to image
        if self.use_landmarks:
            if len(landmarks)==0:
                print("Landmark file not found !")
            landmarks_heatmap = landmark_heatmap(landmarks, image.shape[0], image.shape[1])
            image = np.dstack((image,landmarks_heatmap))
        
        # one-hot encode mask for multiclass
        classes = np.unique(mask)
        if len(classes) > 2:
            masks = [(mask == c) for c in (1,2,3)]
            mask  = np.stack(masks, axis=-1).astype('float')
        
        # transform to tensor format
        sample      = get_tensor()(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']
        
        return image, mask

    def __len__(self):
        return len(self.images_dir)

# TODO : add Compose ?
def get_training_augmentation():
    """Transformation for training set
    input : 
        - input_size : integer for resizing maximum side
    output :
        - transform : object albumentations.Compose
    """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=(-0.5, 0.5), rotate_limit=[-15,15], shift_limit=[-0.1,0.1], p=1, border_mode=0),
        albu.IAAPerspective(p=0.2),
        albu.OneOf([albu.CLAHE(p=1), albu.RandomBrightness(p=1), albu.RandomGamma(p=1)], p=0.3),
        albu.OneOf([albu.Blur(blur_limit=10, p=1), albu.MotionBlur(blur_limit=10, p=1)], p=0.3),
        albu.OneOf([albu.RandomContrast(p=1), albu.HueSaturationValue(p=1)], p=0.2),
        albu.OneOf([albu.IAAAdditiveGaussianNoise(), albu.GaussNoise()], p=0.2),
        albu.HueSaturationValue(p=0.2),
        albu.ISONoise(p=0.1),
        albu.RGBShift(p=0.1),
        #albu.OneOf([albu.OpticalDistortion(p=0.3,distort_limit=0.02),albu.GridDistortion(p=0.2,distort_limit=0.15), albu.IAAPiecewiseAffine(p=0.3, scale=(0.01, 0.01))])
    ]
    return albu.Compose(train_transform)

class OnlyOnMask(albu.Sequential):
    def __call__(self, **data):
        image = data['image'].copy()
        data  = super(OnlyOnMask, self).__call__(**data)
        mask  = data['mask'].astype(np.bool)
        image[mask]   = data['image'][mask]
        data['image'] = image
        return data


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

def to_tensor(x, **kwargs):
    """Return input in tensor format"""
    if len(x.shape)==2:
        x = np.expand_dims(x, 2)
    return x.transpose(2, 0, 1).astype('float32')

def get_tensor():
    """Transform to tensor"""
    return albu.Lambda(image=to_tensor, mask=to_tensor)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    input : 
        - preprocessing_fn : preprocessing function
    output :
        - transform : albumentations object
    """
    return albu.Lambda(image=preprocessing_fn)


def gaussian_2D(pt, sigma, width, height): 
    
    img = np.zeros((width,height))
    
    # Check that any part of the gaussian is in-bounds
    ul = [round(pt[0] - 3 * sigma), round(pt[1] - 3 * sigma)]
    br = [round(pt[0] + 3 * sigma + 1), round(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x    = np.arange(0, size+1, 1, float) # sample the image
    y    = x[:, np.newaxis]
    x0   = size // 2
    y0   = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (sigma ** 2))#(2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])
    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    
    return img

def landmark_heatmap(landmarks, width, height):
    
    gaussians   = np.zeros((width,height,len(landmarks)))
    sigma       = 0.05*max(width, height)//2
    # generate gaussian for each landmark
    for i, landmark in enumerate(landmarks):
        gaussians[:,:,i] = gaussian_2D(landmark, sigma, width, height)
    # get max
    heatmap = np.amax(gaussians, axis=2)
    
    return heatmap

# fix seed
random.seed(12345)
np.random.seed(12345)

def split_data(data_root, image_paths, img_ext, mask_paths, msk_ext, size_dataset=100, use_id=False, train_factor=0.8, valid_factor=0.1, test_factor=0.1):
    # fix seed
    random.seed(12345)
    np.random.seed(12345)     
    # path
    folder_image = glob.glob(os.path.join(data_root, image_paths)+"\\*"+img_ext)
    folder_mask  = glob.glob(os.path.join(data_root, mask_paths,)+"\\*"+msk_ext)
    if not use_id:
        # suffle the 2 lists the same way (to be sure)
        lists_shuffled = list(zip(folder_image, folder_mask))
        random.shuffle(lists_shuffled)
        folder_image, folder_mask = zip(*lists_shuffled)
        # get size
        train_size = int(train_factor * size_dataset)
        valid_size = int(valid_factor * size_dataset)
        test_size  = int(test_factor * size_dataset)
        # split in train/val/test
        train_image = folder_image[:train_size]
        train_mask  = folder_mask[:train_size]
        valid_image = folder_image[train_size:train_size+valid_size]
        valid_mask  = folder_mask[train_size:train_size+valid_size]
        test_image  = folder_image[train_size+valid_size:]
        test_mask   = folder_mask[train_size+valid_size:]
        # print infos
        print('TOTAL :',len(folder_image),' images')
        print(' - train :',len(train_image),' images')
        print(' - valid :',len(valid_image),' images')
        print(' - test  :',len(test_image),' images')
    else:
        # create dictionnary according to name
        image_list = {}
        for image_ref in folder_image:
            image_id  = os.path.basename(image_ref)[:3]
            subset_id = []
            for image in folder_image:
                if image_id in os.path.basename(image):
                    subset_id.append(os.path.basename(image).replace('.png','').replace('.jpg',''))
                    image_list[image_id] = subset_id
        # suffle id list
        image_list_id = list(image_list.keys())
        random.shuffle(image_list_id)
        # get size
        size_dataset = len(image_list)
        train_size = int(train_factor * size_dataset)
        valid_size = int(valid_factor * size_dataset)
        test_size  = int(test_factor * size_dataset)
        # split in train/val/test
        train_id     = image_list_id[:train_size]
        valid_id     = image_list_id[train_size:train_size+valid_size]
        test_id      = image_list_id[train_size+valid_size:]
        # Get back full path
        train_list = []
        for i in range(len(train_id)):
            train_list.extend(image_list[train_id[i]])
        train_image = [os.path.join(data_root, image_paths, image_name+img_ext) for image_name in train_list]
        train_mask  = [os.path.join(data_root, mask_paths, mask_name+msk_ext) for mask_name in train_list]
        valid_list = []
        for i in range(len(valid_id)):
            valid_list.extend(image_list[valid_id[i]])
        valid_image = [os.path.join(data_root, image_paths, image_name+img_ext) for image_name in valid_list]
        valid_mask  = [os.path.join(data_root, mask_paths, mask_name+msk_ext) for mask_name in valid_list]
        test_list = []
        for i in range(len(test_id)):
            test_list.extend(image_list[test_id[i]])
        test_image = [os.path.join(data_root, image_paths, image_name+img_ext) for image_name in test_list]
        test_mask  = [os.path.join(data_root, mask_paths, mask_name+msk_ext) for mask_name in test_list]         
        # print infos
        print('TOTAL :',len(folder_image),' images - ',len(image_list_id),' personnes')
        print('train :',len(train_image),' images - ',len(train_id),' personnes')
        print('valid :',len(valid_image),' images - ',len(valid_id),' personnes')
        print('test  :',len(test_image),' images - ',len(test_id),' personnes')
        
    train_set = (train_image, train_mask)
    valid_set = (valid_image, valid_mask)
    test_set  = (test_image, test_mask)
    
    return train_set, valid_set, test_set
    