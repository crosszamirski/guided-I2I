import os
import sys
sys.path.append('/projects/')
import pandas as pd
import numpy as np
import cv2

from PIL import Image
import os
import shutil


# Set the directory you want to start from
rootDir = '/projects/img/GAN_CP/PAPER_3/Palette-Image-to-Image-Diffusion-Models-main/experiments/test_Target2_230112_105235/results/test/0/'

#rootDir = '/projects/img/GAN_CP/PAPER_3/Palette-Image-to-Image-Diffusion-Models-main/GT-files/'

# Create a new directory to move the files to
destDir = '/projects/img/GAN_CP/PAPER_3/Palette-Image-to-Image-Diffusion-Models-main/Plate_B/AdaGN-Target_CG-None/'
#if not os.path.exists(destDir):
#    os.mkdir(destDir)


def standardize_image(image_in,pixel_cutoff):
        image_in = np.array(image_in)
#        print(image_in)
        image_in = image_in.astype('float32')
        means = image_in.mean(dtype='float64')
        stds = image_in.std(dtype='float64')
#        print('Means: %s, Stds: %s' % (means, stds))
        # per-channel standardization of pixels
        image_in = (image_in - means) / stds
#        # confirm it had the desired effect
#        means = image_in.mean(axis=(0,1), dtype='float64')
#        stds = image_in.std(axis=(0,1), dtype='float64')
#        print('Means: %s, Stds: %s' % (means, stds))      
        image_in[image_in > pixel_cutoff] = pixel_cutoff
        image_in[image_in < -pixel_cutoff] = -pixel_cutoff
        image_in = np.array(image_in)
        return(image_in)



## Loop through all the files in the starting directory
#for dirName, subdirList, fileList in os.walk(rootDir):
#    for fname in fileList:
#        # Check if the file name starts with "GT_"
#        if fname.startswith('GT_'):
#            # Construct the full path of the file
#            srcPath = os.path.join(dirName, fname)
#            destPath = os.path.join(destDir, fname)
#            # Move the file
#            shutil.move(srcPath, destPath)

for file in os.listdir(rootDir):
    # check if the file is a .tiff image
#    print(file)
    # read the image
    img = Image.open(os.path.join(rootDir, file))
#    print(img)
    # normalize the image to have a mean of 0 and standard deviation of 1
    img = standardize_image(img,pixel_cutoff=15)
    
    if file.startswith('Out_'):
      new_name = file[4:-14]
      channel = str(int(file[-5])+1)
      # add "C01.tiff" to the name
      new_name += "A0"
      new_name += channel
      new_name += "Z01C0"
      new_name += channel
      new_name += ".tiff"
      print(new_name)
      cv2.imwrite(os.path.join(destDir, new_name), img)
##        file_new_name = file[:-12]
##        file_new_name += ".tiff"
#        print(file_new_name)
#        new_name = filename[4:-13]
  
  
#  
##  print(new_name)
#  # rename the file
#  os.rename(file_path, os.path.join(folder_path, new_name))





# path to the folder containing the files
#folder_path = '/projects/img/GAN_CP/PAPER_3/Palette-Image-to-Image-Diffusion-Models-main/sample_predicted'#GT_files/'
#

# iterate over all the files in the folder
#for filename in os.listdir(folder_path):
  # get the full path of the file
#  file_path = os.path.join(folder_path, filename)
#  
#  # remove the first 4 and last 13 characters from the name
#  new_name = filename[4:-13]
#  channel = str(int(filename[-5])+1)
#  # add "C01.tiff" to the name
#  new_name += "A0"
#  new_name += channel
#  new_name += "Z01C0"
#  new_name += channel
#  new_name += ".tiff"
#  
#  
##  print(new_name)
#  # rename the file
#  os.rename(file_path, os.path.join(folder_path, new_name))
#
#
#
#
#
## directory containing the .tiff images
#img_dir = '/scratch/os_images/Darwin-cv8000-2021-06/Guy_Target2/resized/1086292884/'
## directory to save the normalized images'
##\\samba.scp.astrazeneca.net\scratch\os_images\Darwin-cv8000-2021-06\Guy_Target2\resized\1086293492
## directory to save the normalized images
#save_dir_CP = '/projects/img/GAN_CP/PAPER_3/Palette-Image-to-Image-Diffusion-Models-main/Plate_B/Ground_Truth/'
#save_dir_BF = '/projects/img/GAN_CP/PAPER_3/Palette-Image-to-Image-Diffusion-Models-main/Plate_B_Brightfield/'
#
#
## create the save directory if it doesn't exist
#if not os.path.exists(save_dir_CP):
#    os.makedirs(save_dir_CP)
#
#if not os.path.exists(save_dir_BF):
#    os.makedirs(save_dir_BF)
#
## loop through all the files in the img_dir directory
#for file in os.listdir(img_dir):
#    # check if the file is a .tiff image
##    print(file)
#    # read the image
#    img = Image.open(os.path.join(img_dir, file))
##    print(img)
#    # normalize the image to have a mean of 0 and standard deviation of 1
#    img = standardize_image(img,pixel_cutoff=15)
#    
#    if file.endswith('C06_resize.tiff'):
#        pass
#        # save the normalized image
##        file_new_name = file[:-12]
##        file_new_name += ".tiff"
##        print(file_new_name)
##        cv2.imwrite(os.path.join(save_dir_BF, file_new_name), img)
#    else:
#        file_new_name = file[:-12]
#        file_new_name += ".tiff"
#        print(file_new_name)
#        cv2.imwrite(os.path.join(save_dir_CP, file_new_name), img)
#
#




