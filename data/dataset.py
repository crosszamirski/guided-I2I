import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
import albumentations
import pandas as pd
import cv2
import random

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)






class Target2Dataset(data.Dataset):
    def __init__(self, path10):
        path0 = pd.read_csv(path10)
        self.pixel_cutoff = 15
        channel_headers = ['C01','C02','C03','C04','C05','C06_1','C06_2','C06_3']
        weak_label_headers = ['Unique_Pert', 'Unique_Plate', 'Unique_Target']
        
        self.X01 = path0[channel_headers[0]] 
        self.X02 = path0[channel_headers[1]]
        self.X03 = path0[channel_headers[2]]
        self.X04 = path0[channel_headers[3]]
        self.X05 = path0[channel_headers[4]]
        self.X06_1 = path0[channel_headers[5]]
        self.X06_2 = path0[channel_headers[6]]
        self.X06_3 = path0[channel_headers[7]]
                
        self.tag = path0[weak_label_headers[0]]
        self.tag2 = path0[weak_label_headers[1]]
        self.tag3 = path0[weak_label_headers[2]]
        
        self.aug0 = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.augmentations.geometric.rotate.RandomRotate90(p = 0.5),
#        albumentations.Normalize(mean=[0],std=[1],max_pixel_value=self.pixel_cutoff, always_apply=True),
#        albumentations.augmentations.crops.transforms.RandomCrop (256, 256, always_apply=True),
        #albumentations.augmentations.crops.transforms.RandomResizedCrop(512,512, scale=(0.9,1.0), ratio=(1,1), interpolation=cv2.INTER_CUBIC,always_apply=True)
        ],
        additional_targets={'image1':'image','image2':'image','image3':'image','image4':'image','image5':'image','image6':'image','image7':'image'})

    
    
    def standardize_image(self, image_in):
        image_in = np.array(image_in)
        image_in = image_in.astype('float32')
        means = image_in.mean(axis=(0,1), dtype='float64')
        stds = image_in.std(axis=(0,1), dtype='float64')
#        print('Means: %s, Stds: %s' % (means, stds))
        # per-channel standardization of pixels
        image_in = (image_in - means) / stds
#        # confirm it had the desired effect
#        means = image_in.mean(axis=(0,1), dtype='float64')
#        stds = image_in.std(axis=(0,1), dtype='float64')
#        print('Means: %s, Stds: %s' % (means, stds))        
        image_in = np.array(image_in)
        image_in[image_in > self.pixel_cutoff] = self.pixel_cutoff
        image_in[image_in < -self.pixel_cutoff] = -self.pixel_cutoff
        return(image_in)
                
        
    def __len__(self):
        return (len(self.X01))

    def __getitem__(self, i):
        
        ret = {}
        label = self.tag[i]
        label2 = self.tag2[i]
        label3 = self.tag3[i]
#        same_idx = self.tag[self.tag == Label].index.values
        
        Aimage = Image.open(self.X01[i])
        Aimage1 = Image.open(self.X02[i])
        Aimage2 = Image.open(self.X03[i])
        Aimage3 = Image.open(self.X04[i])
        Aimage4 = Image.open(self.X05[i])
        Aimage5 = Image.open(self.X06_1[i])
        Aimage6 = Image.open(self.X06_2[i])
        Aimage7 = Image.open(self.X06_3[i])
        
        Aimage = self.standardize_image(Aimage)
        Aimage1 = self.standardize_image(Aimage1)
        Aimage2 = self.standardize_image(Aimage2)
        Aimage3 = self.standardize_image(Aimage3)
        Aimage4 = self.standardize_image(Aimage4)
        Aimage5 = self.standardize_image(Aimage5)
        Aimage6 = self.standardize_image(Aimage6)
        Aimage7 = self.standardize_image(Aimage7)
        
        transformed0 = self.aug0(image=Aimage, image1=Aimage1, image2=Aimage2, 
                                 image3=Aimage3, image4 =Aimage4, image5=Aimage5, 
                                 image6=Aimage6, image7=Aimage7)
        image = transformed0['image']
        image_0 = image.astype(np.float32)
        image1 = transformed0['image1']
        image1_0 = image1.astype(np.float32)
        image2 = transformed0['image2']
        image2_0 = image2.astype(np.float32)
        image3 = transformed0['image3']
        image3_0 = image3.astype(np.float32)
        image4 = transformed0['image4']
        image4_0 = image4.astype(np.float32)
        image5 = transformed0['image5']
        image5_0 = image5.astype(np.float32)
        image6 = transformed0['image6']
        image6_0 = image6.astype(np.float32)
        image7 = transformed0['image7']
        image7_0 = image7.astype(np.float32)


        image_0 = np.expand_dims(image_0,0)
        image_1 = np.expand_dims(image1_0,0)
        image_2 = np.expand_dims(image2_0,0)
        image_3 = np.expand_dims(image3_0,0)
        image_4 = np.expand_dims(image4_0,0)
        image_5 = np.expand_dims(image5_0,0)      
        image_6 = np.expand_dims(image6_0,0)
        image_7 = np.expand_dims(image7_0,0)
             
        image_0 = np.concatenate((image_0, image_1, image_2, image_3, image_4),axis = 0)
        image_1 = np.concatenate((image_5, image_6, image_7),axis = 0)
                
        img = torch.tensor(image_0, dtype=torch.float)
#        img = torch.sub(img,0.5)
#        img = torch.multiply(img,2)
        cond_img = torch.tensor(image_1, dtype=torch.float) 
#        cond_img = torch.sub(cond_img,0.5)
#        cond_img = torch.multiply(cond_img,2)

        ret['gt_image'] = img # (output) Cell Painting 5x
        ret['cond_image'] = cond_img # (input) Brightfield 3x
        path = self.X01[i]
        pathy = path[-48:-21]
        ret['path'] = pathy
        ret['class'] = label# pert
        ret['plate'] = label2# BATCH/PLATE
        ret['target'] = label3# TARGET
        
        if label3 == 145:
            ret["DMSO"] = 0
        else:
            ret["DMSO"] = 1
#        print(out_dict["y"])
        
        return ret






class Target2Dataset_test(data.Dataset):
    def __init__(self, path10):
        path0 = pd.read_csv(path10)
        self.pixel_cutoff = 15 # since std of each channel for each image is set to 1 then this will remove the >99.7% pixel outliers
        channel_headers = ['C01','C02','C03','C04','C05','C06_1','C06_2','C06_3']
        weak_label_headers = ['Unique_Pert', 'Unique_Plate','Unique_Target']
        
        self.X01 = path0[channel_headers[0]] 
        self.X02 = path0[channel_headers[1]]
        self.X03 = path0[channel_headers[2]]
        self.X04 = path0[channel_headers[3]]
        self.X05 = path0[channel_headers[4]]
        self.X06_1 = path0[channel_headers[5]]
        self.X06_2 = path0[channel_headers[6]]
        self.X06_3 = path0[channel_headers[7]]
                
        self.tag = path0[weak_label_headers[0]]
        self.tag2 = path0[weak_label_headers[1]]
        self.tag3 = path0[weak_label_headers[2]]
        
        self.aug0 = albumentations.Compose([
#        albumentations.HorizontalFlip(p=0.5),
#        albumentations.VerticalFlip(p=0.5),
#        albumentations.Normalize(mean=[0],std=[1],max_pixel_value=self.pixel_cutoff, always_apply=True),
#        albumentations.augmentations.crops.transforms.RandomCrop (256, 256, always_apply=True),
        ],
        additional_targets={'image1':'image','image2':'image','image3':'image','image4':'image','image5':'image','image6':'image','image7':'image'})

    def standardize_image(self, image_in):
        image_in = np.array(image_in)
        image_in = image_in.astype('float32')
        means = image_in.mean(axis=(0,1), dtype='float64')
        stds = image_in.std(axis=(0,1), dtype='float64')
#        print('Means: %s, Stds: %s' % (means, stds))
        # per-channel standardization of pixels
        image_in = (image_in - means) / stds
#        # confirm it had the desired effect
#        means = image_in.mean(axis=(0,1), dtype='float64')
#        stds = image_in.std(axis=(0,1), dtype='float64')
#        print('Means: %s, Stds: %s' % (means, stds))        
        image_in = np.array(image_in)
        image_in[image_in > self.pixel_cutoff] = self.pixel_cutoff
        image_in[image_in < -self.pixel_cutoff] = -self.pixel_cutoff
        return(image_in)
            
        
    
    def __len__(self):
        return (len(self.X01))

    def __getitem__(self, i):
        
        ret = {}
        label = self.tag[i]
        label2 = self.tag2[i]
        label3 = self.tag3[i]
#        same_idx = self.tag[self.tag == Label].index.values
        
        Aimage = Image.open(self.X01[i])
        Aimage1 = Image.open(self.X02[i])
        Aimage2 = Image.open(self.X03[i])
        Aimage3 = Image.open(self.X04[i])
        Aimage4 = Image.open(self.X05[i])
        Aimage5 = Image.open(self.X06_1[i])
        Aimage6 = Image.open(self.X06_2[i])
        Aimage7 = Image.open(self.X06_3[i])
        
        Aimage = self.standardize_image(Aimage)
        Aimage1 = self.standardize_image(Aimage1)
        Aimage2 = self.standardize_image(Aimage2)
        Aimage3 = self.standardize_image(Aimage3)
        Aimage4 = self.standardize_image(Aimage4)
        Aimage5 = self.standardize_image(Aimage5)
        Aimage6 = self.standardize_image(Aimage6)
        Aimage7 = self.standardize_image(Aimage7)
        
        transformed0 = self.aug0(image=Aimage, image1=Aimage1, image2=Aimage2, 
                                 image3=Aimage3, image4 =Aimage4, image5=Aimage5, 
                                 image6=Aimage6, image7=Aimage7)
        image = transformed0['image']
        image_0 = image.astype(np.float32)
        image1 = transformed0['image1']
        image1_0 = image1.astype(np.float32)
        image2 = transformed0['image2']
        image2_0 = image2.astype(np.float32)
        image3 = transformed0['image3']
        image3_0 = image3.astype(np.float32)
        image4 = transformed0['image4']
        image4_0 = image4.astype(np.float32)
        image5 = transformed0['image5']
        image5_0 = image5.astype(np.float32)
        image6 = transformed0['image6']
        image6_0 = image6.astype(np.float32)
        image7 = transformed0['image7']
        image7_0 = image7.astype(np.float32)


        image_0 = np.expand_dims(image_0,0)
        image_1 = np.expand_dims(image1_0,0)
        image_2 = np.expand_dims(image2_0,0)
        image_3 = np.expand_dims(image3_0,0)
        image_4 = np.expand_dims(image4_0,0)
        image_5 = np.expand_dims(image5_0,0)      
        image_6 = np.expand_dims(image6_0,0)
        image_7 = np.expand_dims(image7_0,0)
             
        image_0 = np.concatenate((image_0, image_1, image_2, image_3, image_4),axis = 0)
        image_1 = np.concatenate((image_5, image_6, image_7),axis = 0)
                
        img = torch.tensor(image_0, dtype=torch.float)
#        img = torch.subtract(img,0.5)
#        img = torch.multiply(img,2)
        cond_img = torch.tensor(image_1, dtype=torch.float) 
#        cond_img = torch.subtract(cond_img,0.5)
#        cond_img = torch.multiply(cond_img,2)

        ret['gt_image'] = img # (output) Cell Painting 5x
        ret['cond_image'] = cond_img # (input) Brightfield 3x
        path = self.X01[i]
        pathy = path[-48:-21]
        ret['path'] = pathy
#        print(ret['path'])
        ret['class'] = label# WEAK LABEL
        ret['plate'] = label2# BATCH/PLATE
        ret['target'] = label3 # TARGET
        
        if label3 == 145:
            ret["DMSO"] = 0
        else:
            ret["DMSO"] = 1
            
        return ret

