import numpy as np
import os
import tifffile as tiff
import random
from matplotlib import pyplot as plt
from PIL import Image
from e_run import Parameters


img_tiff_path = './d_data/SOFC_anode_segmented.tif'


flag = 'train'

channel_depth = Parameters.L
img_size = (channel_depth, Parameters.img_size, Parameters.img_size)

if flag=='train':
    N_num = 2000
else:
    N_num = 200
count=0

img_output_path = f'./d_data/Stage2_{img_size[0]}x{img_size[1]}x{img_size[2]}_tiff_' + flag


os.makedirs(img_output_path, exist_ok=True)


img = tiff.imread(img_tiff_path)

print(f'img.shape: {img.shape}')

dim_1_limit = 1500 # seperate the data sample for train and validation.

for i in range(1000000):

    print(f'i={i}')
    i_r = random.randint(0, img.shape[0]-img_size[0])
    j_r = random.randint(0, img.shape[1]-img_size[1])
    k_r = random.randint(0, img.shape[2]-img_size[2])
    if flag == 'train':
        condition  = k_r < dim_1_limit
        print(condition)
    else:
        condition  = k_r >= dim_1_limit
    if condition:
        img_crop = img[i_r:i_r+img_size[0], j_r:j_r+img_size[1], k_r:k_r+img_size[2]]
        tiff.imwrite(img_output_path + '/'+ f'mixed_MPL_stack_{img_size[0]}x{img_size[1]}x{img_size[2]}_' +str(count)+ '.tif', img_crop, imagej=True, bigtiff=False)
        count +=1
        print(f'Processing image {count}')

    if count == N_num:

        break




