import math
import random
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import blobfile as bf
#from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import torch
import albumentations
import pandas as pd

def load_data(
    *,
    data_dir,
    batch_size,
    deterministic=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    dataset = Target2Dataset(path10 = data_dir)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


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
        
#        ret = {}
#        label = self.tag[i]
#        label2 = self.tag2[i]
        label3 = self.tag3[i] # target
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
             
        image_0 = np.concatenate((image_5, image_6, image_7, image_0, image_1, image_2, image_3, image_4),axis = 0)
#        image_1 = np.concatenate((image_5, image_6, image_7),axis = 0)
        
#        image_2 = np.concatenate((image_1, image_0), axis=0)
                
#        img = torch.tensor(image_0, dtype=torch.float)
#        img = torch.sub(img,0.5)
#        img = torch.multiply(img,2)
#        cond_img = torch.tensor(image_1, dtype=torch.float) 
#        cond_img = torch.sub(cond_img,0.5)
#        cond_img = torch.multiply(cond_img,2)
        
        img = torch.tensor(image_0, dtype=torch.float) 
#        ret['gt_image'] = img # (output) Cell Painting 5x
#        reta['cond_image'] = cond_img # (input) Brightfield 3x
#        path = self.X01[i]
#        pathy = path[-48:-21]
#        reta['path'] = pathy
        out_dict = {}
        out_dict["y"] = label3#3
#        print(label)
#        if label3 == 145:
#            out_dict["y"] = 0
#        else:
#            out_dict["y"] = 1
#        print(out_dict["y"])
#        reta['plate'] = label2# BATCH/PLATE
#        reta['target'] = label3# TARGET
#        print(reta)
        return img, out_dict



#
#class ImageDataset(Dataset):
#    def __init__(
#        self,
#        resolution,
#        image_paths,
#        classes=None,
#        shard=0,
#        num_shards=1,
#        random_crop=False,
#        random_flip=True,
#    ):
#        super().__init__()
#        self.resolution = resolution
#        self.local_images = image_paths[shard:][::num_shards]
#        self.local_classes = None if classes is None else classes[shard:][::num_shards]
#        self.random_crop = random_crop
#        self.random_flip = random_flip
#
#    def __len__(self):
#        return len(self.local_images)
#
#    def __getitem__(self, idx):
#        path = self.local_images[idx]
#        with bf.BlobFile(path, "rb") as f:
#            pil_image = Image.open(f)
#            pil_image.load()
#        pil_image = pil_image.convert("RGB")
#
#        if self.random_crop:
#            arr = random_crop_arr(pil_image, self.resolution)
#        else:
#            arr = center_crop_arr(pil_image, self.resolution)
#
#        if self.random_flip and random.random() < 0.5:
#            arr = arr[:, ::-1]
#
#        arr = arr.astype(np.float32) / 127.5 - 1
#
#        out_dict = {}
#        if self.local_classes is not None:
#            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
#        return np.transpose(arr, [2, 0, 1]), out_dict
#
#
#def center_crop_arr(pil_image, image_size):
#    # We are not on a new enough PIL to support the `reducing_gap`
#    # argument, which uses BOX downsampling at powers of two first.
#    # Thus, we do it by hand to improve downsample quality.
#    while min(*pil_image.size) >= 2 * image_size:
#        pil_image = pil_image.resize(
#            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#        )
#
#    scale = image_size / min(*pil_image.size)
#    pil_image = pil_image.resize(
#        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#    )
#
#    arr = np.array(pil_image)
#    crop_y = (arr.shape[0] - image_size) // 2
#    crop_x = (arr.shape[1] - image_size) // 2
#    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
#
#
#def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
#    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
#    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
#    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)
#
#    # We are not on a new enough PIL to support the `reducing_gap`
#    # argument, which uses BOX downsampling at powers of two first.
#    # Thus, we do it by hand to improve downsample quality.
#    while min(*pil_image.size) >= 2 * smaller_dim_size:
#        pil_image = pil_image.resize(
#            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#        )
#
#    scale = smaller_dim_size / min(*pil_image.size)
#    pil_image = pil_image.resize(
#        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#    )
#
#    arr = np.array(pil_image)
#    crop_y = random.randrange(arr.shape[0] - image_size + 1)
#    crop_x = random.randrange(arr.shape[1] - image_size + 1)
#    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
