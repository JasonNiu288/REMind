import torch
import numpy as np
import os
import tifffile as tiff
import random
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn.functional as F
from e_run import Parameters

img_tiff_path = './d_data/SOFC_anode_segmented.tif'


flag = 'train'   #Switch between 'train' and 'test' to generate train and test datasets

channel_depth = 1
img_size = (channel_depth, Parameters.img_size, Parameters.img_size)
sf = Parameters.sf 
target_size = int(img_size[1]/sf)

if flag=='train':
    N_num = 2000
else:
    N_num = 200
count=0
img_output_path = f'./d_data/Stage1_sf={sf}_{img_size[0]}x{img_size[1]}x{img_size[2]}_tiff_' + flag


os.makedirs(img_output_path, exist_ok=True)


img = tiff.imread(img_tiff_path)

print(f'img.shape: {img.shape}')

dim_1_limit = 1500 # seperate the data sample for train and validation.

count = 0
for i in range(100000):


    i_r = random.randint(0, img.shape[0]-1)
    j_r = random.randint(0, img.shape[1] - img_size[1])
    k_r = random.randint(0, img.shape[2] - img_size[2])
    
    if flag == 'train':
        condition  = k_r < dim_1_limit
        print(condition)
    else:
        condition  = k_r >= dim_1_limit
    if condition:
        img_crop = img[i_r, j_r:j_r+img_size[1], k_r:k_r+img_size[2]]

        #initialise an array to save both HR and LR images
        img_LH = np.zeros((img_size[1],img_size[1] + target_size))
        img_LH[:, 0:img_size[1]] = img_crop

        img_HR = torch.from_numpy(img_crop).unsqueeze(dim=0).unsqueeze(dim=0)
        img_LR = F.interpolate(img_HR, size=(target_size, target_size), mode='bicubic', align_corners=False)
        img_LH[0:target_size, img_size[1]:(img_size[1] + target_size)] = img_LR[0, 0,:]


        tiff.imwrite(img_output_path + '/'+ f'LR_HR_pairs_{img_size[1]}x{img_size[2]}_' +str(count)+ '.tif', img_LH.astype(np.uint8), imagej=True, bigtiff=False)
    
        print(f'Processing image {count}')
        count += 1

    if count == N_num:

        break




