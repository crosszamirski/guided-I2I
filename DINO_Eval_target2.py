import os
import sys
import argparse
import albumentations
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SequentialSampler
from torchvision import models as torchvision_models
from sklearn import preprocessing
from torch.utils.data import Dataset
import utils
import vision_transformer as vits
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
from sklearn.decomposition import PCA as sk_PCA

channels_CP = ['C01','C02','C03','C04','C05']
channels_BF = ['C06_1','C06_2','C06_3']

"Set A Evaluation"


folder_path = "/projects/img/GAN_CP/PAPER_3/Palette-Image-to-Image-Diffusion-Models-main/Plate_A/Ground_Truth/"

x0train = pd.read_csv(f'/projects/img/GAN_CP/PAPER_3/src/target2_SET_A_test_general.csv')
#x0DMSO = pd.read_csv(f'/projects/img/GAN_CP/PAPER_3/src/target2_SET1_CONTROLS.csv')

model_evaluating = "SETA"
#
#"Set 2 Evaluation"
#
#x0train = pd.read_csv(f'/projects/img/GAN_CP/PAPER_3/src/target2_SET2.csv')
#x0DMSO = pd.read_csv(f'/projects/img/GAN_CP/PAPER_3/src/target2_SET2_CONTROLS.csv')
#
#model_evaluating = "SET2"

"Parameters to set:"

#channels_CP2 = ['C02','C03']

channel_headers = channels_CP # channels_CP, channels_BF
weight_type = "imagenet" #PSUEDO_WSDINO" #"imagenet" # "WSDINO"
start_epoch = 0
end_epoch = 10
frequency = 10

pixel_cutoff = 15
num_classes = 145 # number of targets

def extract_feature_pipeline(args, weights,channel):
    dataset_train = ReturnIndexDataset(x0train, channel)
#    dataset_train2 = ReturnIndexDataset_DMSO(x0DMSO, channel)
    sampler = SequentialSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
#    sampler2 = SequentialSampler(dataset_train2)
#    data_loader_train2 = torch.utils.data.DataLoader(
#        dataset_train2,
#        sampler=sampler2,
#        batch_size=args.batch_size_per_gpu,
#        num_workers=args.num_workers,
#        pin_memory=True,
#        drop_last=True,
#    )
    
    print(f"Data loaded with {len(dataset_train)} imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=145)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=145)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=145)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    print("Extracting features from train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print(train_features)
    print(train_features.size())
    
#    print("Extracting features from DMSO set...")
#    train_features2 = extract_features2(model, data_loader_train2, args.use_cuda)
#    print(train_features2)
#    print(train_features2.size())

    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, f"target2_trainfeat.pth"))
        train_features_cpu = train_features.cpu()
        features_np = train_features_cpu.numpy() #convert to Numpy array
        df_csv = pd.DataFrame(features_np) #convert to a dataframe
        df_csv.to_csv("target2_trainfeatures.csv",index=True) #save to file
        
    return train_features#, train_features2#, test_features, train_labels, test_labels

@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index, target, pert, plate in metric_logger.log_every(data_loader, 10):
        index = index.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        pert = pert.cuda(non_blocking=True)
        plate = plate.cuda(non_blocking=True)
        feats = []
        for samp in range(4):
            a = samples[samp]
            a = a.cuda(non_blocking=True)
            if multiscale:
                feats_hold = utils.multi_scale(a, model)
            else:
                feats_hold = model(a).clone()
                
            feats.append(feats_hold)

        feats = torch.median(torch.stack(feats),dim=0)
        feats = feats[0]
        feats = feats.flatten()
        feats = torch.cat((feats,target,pert,plate),0)
        feats = feats.unsqueeze(0)          
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features

def extract_features2(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index in metric_logger.log_every(data_loader, 10):
        index = index.cuda(non_blocking=True)
        feats = []
        for samp in range(4):
            a = samples[samp]
            a = a.cuda(non_blocking=True)
            if multiscale:
                feats_hold = utils.multi_scale(a, model)
            else:
                feats_hold = model(a).clone()
                
            feats.append(feats_hold)

        feats = torch.median(torch.stack(feats),dim=0)
        feats = feats[0]
        feats = feats.flatten()
        feats = feats.unsqueeze(0)            
        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


#def correct_tvn(features_DMSO, features_all):
#    DMSO_features_cpu = features_DMSO.cpu()
#    features_DMSO_np = DMSO_features_cpu.numpy() #convert to Numpy array
#    features_cpu = features_all.cpu()
#    features_all_np = features_cpu.numpy() #convert to Numpy array
#    labels = features_all_np[:,-3:]
#    features_all_np = features_all_np[:,:-3]
#    p = sk_PCA(n_components=384, whiten=True).fit(features_DMSO_np)
#    features_all = p.transform(features_all_np)
#    features_all = np.concatenate([features_all,labels], axis=1)
#    return features_all


def Aggregate_features_NSC(features, channel, epoch):
    features_np = features
    df = pd.DataFrame(features_np)
    df.rename(columns={ df.columns[384]: "target" }, inplace = True)
    df.rename(columns={ df.columns[385]: "pert" }, inplace = True)
    df.rename(columns={ df.columns[386]: "plate" }, inplace = True)
    df = df.groupby(['pert','plate'],as_index=False).mean()
#    print(df)
    df = df.groupby('pert').mean()
#    print(df)
    df = df.drop("plate", axis=1)
    df['pert'] = df.index
    print(df)
    df.to_csv(f"NSC_features_{channel}_model_{weight_type}_epoch_{epoch}.csv",index=True) #save to file
    return df


def Aggregate_features_NSCB(features, channel, epoch):
    features_np = features
    df = pd.DataFrame(features_np)
    df.rename(columns={ df.columns[384]: "target" }, inplace = True)
    df.rename(columns={ df.columns[385]: "pert" }, inplace = True)
    df.rename(columns={ df.columns[386]: "plate" }, inplace = True)
    df = df.groupby(['pert','plate'],as_index=False).mean()
    print(df)
#    df = df.groupby('pert').mean()
#    df = df.drop("plate", axis=1)
    df.to_csv(f"NSCB_features_{channel}_model_{weight_type}_epoch_{epoch}.csv",index=True) #save to file
    return df


      
def NSCB_function(features, channel, epoch):
    df = Aggregate_features_NSCB(features, channel, epoch)
    df = pd.DataFrame(df)
    label_df = df[["pert", "target", "plate"]]
    feature_df = df.iloc[: , :-3]
#    feature_df = preprocessing.normalize(feature_df, norm='l2')
    feature_df = pd.DataFrame(feature_df)
    print(feature_df)
    print(label_df)
    tally = []
    for idx in range(len(label_df)):
        print(idx) # index
        feature = feature_df.iloc[[idx]] # feature vector
        same_compound_feat = label_df.iloc[[idx]]
        same_compound_val = same_compound_feat[["pert"]]
        same_compound_val = same_compound_val.to_numpy()
        same_compound_val = same_compound_val.item(0)
        same_batch_val = same_compound_feat[["plate"]]
        same_batch_val = same_batch_val.to_numpy()
        same_batch_val = same_batch_val.item(0)
        drop_index1 = label_df.loc[label_df['pert'] == same_compound_val]
        remaining_features = feature_df.drop(drop_index1.index)
        label_df_dropped = label_df.drop(drop_index1.index)        
        drop_index2 = label_df_dropped.loc[label_df_dropped['plate'] == same_batch_val]
        label_df_dropped = label_df_dropped.drop(drop_index2.index)
        remaining_features1 = remaining_features.drop(drop_index2.index)
        remaining_features = remaining_features1     
        remaining_features['cos_sim'] = cosine_similarity(remaining_features, feature).reshape(-1)
        nn = remaining_features[['cos_sim']].idxmax()        
        dif_compound_val = label_df_dropped.loc[nn]
        print('dif pert')
        print(dif_compound_val)
        moa_dif = dif_compound_val[["target"]]
        moa_dif = moa_dif.to_numpy()
        moa_dif = moa_dif.item(0)

        moa_orig = same_compound_feat[["target"]]
        moa_orig = moa_orig.to_numpy()
        moa_orig = moa_orig.item(0)
        print('target_orig')
        print(moa_orig)
        print('target_dif')
        print(moa_dif)
               
        if moa_orig == moa_dif:
            tally.append(1)
        else:
            tally.append(0)
        a_ret = np.mean(tally)
        print(a_ret)
    return a_ret


def NSC_function(features, channel, epoch):
    df = Aggregate_features_NSC(features, channel, epoch)
    label_df = df[["pert", "target"]]
    feature_df = df.iloc[: , :-2]
#    feature_df = preprocessing.normalize(feature_df, norm='l2')
    feature_df = pd.DataFrame(feature_df)
    print(feature_df)
    print(label_df)
    tally = []
    for idx in range(len(label_df)):
        print(idx) # index
        feature = feature_df.iloc[[idx]] # feature vector
        same_compound_feat = label_df.iloc[[idx]]
        same_compound_val = same_compound_feat[["pert"]]
        same_compound_val = same_compound_val.to_numpy()
        same_compound_val = same_compound_val.item(0)
        print('feature')
        print(feature)
        drop_index = label_df.loc[label_df['pert'] == same_compound_val]
        drop_index = drop_index.index
        remaining_features1 = feature_df.drop(drop_index)
        remaining_features = remaining_features1#
        remaining_features = remaining_features.reset_index(drop=True)
     
        remaining_features['cos_sim'] = cosine_similarity(remaining_features, feature).reshape(-1)
        nn = remaining_features[['cos_sim']].idxmax()
        label_df_dropped = label_df.drop(drop_index)
        dif_compound_val = label_df_dropped.iloc[nn]
        print('dif compound')
        print(dif_compound_val)
        moa_dif = dif_compound_val[["target"]]
        moa_dif = moa_dif.to_numpy()
        moa_dif = moa_dif.item(0)

        moa_orig = same_compound_feat[["target"]]
        moa_orig = moa_orig.to_numpy()
        moa_orig = moa_orig.item(0)
        print('moa_orig')
        print(moa_orig)
        print('moa_dif')
        print(moa_dif)
               
        if moa_orig == moa_dif:
            tally.append(1)
        else:
            tally.append(0)
        a_ret = np.mean(tally)
        print(a_ret)
    return a_ret  

class ReturnIndexDataset(Dataset):
    def __init__(self, path0, channel):
        self.X0 = folder_path + path0[channel]
        
        self.y_target = path0['Unique_Target']
        self.y_pert = path0['Unique_Pert']
        self.y_plate = path0['Unique_Plate']
           
#        self.aug0 = albumentations.Compose([
#        albumentations.Normalize(mean=[0],std=[1],max_pixel_value=pixel_cutoff, always_apply=True),])
        self.aug1 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=32, y_min=32, x_max=256, y_max=256, always_apply=True),])
        self.aug2 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=256, y_min=32, x_max=480, y_max=256, always_apply=True),])
        self.aug3 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=32, y_min=256, x_max=256, y_max=480, always_apply=True),])
        self.aug4 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=256, y_min=256, x_max=480, y_max=480, always_apply=True),])
    
    
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
        image_in[image_in > pixel_cutoff] = pixel_cutoff
        image_in[image_in < -pixel_cutoff] = -pixel_cutoff
        return(image_in)
        
    
    def __len__(self):
        return (len(self.X0))  
        
    def __getitem__(self,idx):
#        print('path')
#        print(self.X0[idx])
        Aimage = Image.open(self.X0[idx])
        Aimage = self.standardize_image(Aimage)
        image_0 = Aimage.astype(np.float32)
        crops = []
        
        transformed1 = self.aug1(image=image_0)
        transformed2 = self.aug2(image=image_0)
        transformed3 = self.aug3(image=image_0)
        transformed4 = self.aug4(image=image_0)
        
        image1 = transformed1['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed2['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed3['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed4['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        target = self.y_target[idx]
        pert = self.y_pert[idx]
        plate = self.y_plate[idx]
        
        return crops, idx, target, pert, plate

class ReturnIndexDataset_DMSO(Dataset):
    def __init__(self, path0, channel):
        
        self.X0 = path0[channel]                   
#        self.aug0 = albumentations.Compose([
#        albumentations.Normalize(mean=[0],std=[1],max_pixel_value=pixel_cutoff, always_apply=True),])
        self.aug1 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=32, y_min=32, x_max=256, y_max=256, always_apply=True),])
        self.aug2 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=256, y_min=32, x_max=480, y_max=256, always_apply=True),])
        self.aug3 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=32, y_min=256, x_max=256, y_max=480, always_apply=True),])
        self.aug4 = albumentations.Compose([
        albumentations.augmentations.crops.transforms.Crop(x_min=256, y_min=256, x_max=480, y_max=480, always_apply=True),])
    
    
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
        image_in[image_in > pixel_cutoff] = pixel_cutoff
        image_in[image_in < -pixel_cutoff] = -pixel_cutoff
        return(image_in)
    
    
    def __len__(self):
        return (len(self.X0))  
        
    def __getitem__(self,idx):
        Aimage = Image.open(self.X0[idx])
        Aimage = self.standardize_image(Aimage)
        image_0 = Aimage.astype(np.float32)
        crops = []
        
        transformed1 = self.aug1(image=image_0)
        transformed2 = self.aug2(image=image_0)
        transformed3 = self.aug3(image=image_0)
        transformed4 = self.aug4(image=image_0)
        
        image1 = transformed1['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed2['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed3['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        image1 = transformed4['image']
        image_01 = image1.astype(np.float32)
        image_01 = np.expand_dims(image_01,0)
        image_01 = np.concatenate((image_01, image_01, image_01), axis=0)
        image_01 = torch.tensor(image_01, dtype=torch.float)
        crops.append(image_01)
        
        return crops, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[1], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.04, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default='/projects/img/GAN_CP/PAPER_3/src/Features_for_each_model/script_outputs/',
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    args = parser.parse_args()
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    tally_epoch_nsc = []
    tally_epoch_nscb = []
    
    df_to_save = pd.DataFrame(columns=channel_headers)
    print(df_to_save)
    
    
    for channel in channel_headers:
        print(channel)
        for train_epoch in range(start_epoch,end_epoch,frequency):
            if weight_type == "imagenet":
                weights = 'pretrain_full_checkpoint.pth'
            else:
#                weights = f'{channel}_WSDINO_checkpoint{train_epoch}.pth'
                weights = f'{channel}_PSUEDO_WSDINO_checkpoint{train_epoch}.pth'


            train_features = extract_feature_pipeline(args,weights,channel)
            if utils.get_rank() == 0:
                if args.use_cuda:
                    train_features = train_features.cuda()
#                    DMSO_features = DMSO_features.cuda()
            
#            train_features = train_features#correct_tvn(DMSO_features, train_features)
#            nscb_epoch = NSCB_function(train_features, channel, train_epoch)
#            tally_epoch_nscb.append(nscb_epoch)
#            print(tally_epoch_nscb)
            train_features = train_features.cpu()
            train_features = train_features.numpy()
            print(train_features)
            nsc_epoch = NSC_function(train_features, channel, train_epoch)
            df_to_save.loc[train_epoch, channel] = nsc_epoch
#            tally_epoch_nsc.append(nsc_epoch)
            print(df_to_save)
#            print(tally_epoch_nsc)
    df_to_save.to_csv(f"NSC_set_{model_evaluating}_model_{weight_type}_channels_{channel_headers}.csv",index=True) #save to file
    
    dist.barrier()
    
    

    
    
    
    
    
    
    
    
