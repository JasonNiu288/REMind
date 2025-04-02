import numpy as np
import tifffile as tiff
import os
from tqdm import tqdm







def adjacent_cells(img, cell_index):
    x0 = cell_index[0]
    y0 = cell_index[1]
    z0 = cell_index[2]
    
    adjacent_cells = np.empty([6,1])
   # cell bottom 1
    if  x0 - 1 >=0:   
        adjacent_cells[0,] = img[x0 - 1, y0, z0] 
    else:
        adjacent_cells[0,] = -2 
   # cell top 2
    if  x0 + 1 >=img.shape[0]:
        adjacent_cells[1,] = -2
    else:
        adjacent_cells[1,] = img[x0 + 1, y0, z0] 


   # cell left 3
    if  y0 - 1 >=0:
        adjacent_cells[2,] = img[x0, y0 - 1, z0] 
    else:
        adjacent_cells[2,] = -2 
   # cell right 4
    if  y0 + 1 >=img.shape[1]:
        adjacent_cells[3,] = -2
    else:
        adjacent_cells[3,] = img[x0, y0 + 1, z0] 


   # cell left 5
    if  z0 - 1 >=0:
        adjacent_cells[4,] = img[x0, y0, z0 - 1] 
    else:
        adjacent_cells[4,] = -2 
   # cell right 6
    if  z0 + 1 >=img.shape[2]:
        adjacent_cells[5,] = -2
    else:
        adjacent_cells[5,] = img[x0, y0, z0+ 1] 


    return adjacent_cells


def delete_black_spots(img, pixel_value_carbon, pixel_value_electrolyte, pixel_value_pore):

    for i in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):

                index = np.array([i,j,k])
                neighbour_pixel = adjacent_cells(img, index)

                # Compute unique values and their frequencies
                unique_values, counts = np.unique(neighbour_pixel, return_counts=True)

                # Find the value with the maximum count
                max_count_index = np.argmax(counts)
                min_count_index = np.argmin(counts)
                value_with_max_count = unique_values[max_count_index]
                value_with_min_count = unique_values[min_count_index]
                if (img[i, j, k] in unique_values) == False:
                    print(f'{img[i, j, k]}' 'to' f'{value_with_max_count}')
                    img[i, j, k] = value_with_max_count

                if (img[i, j, k] in unique_values):
                    idx_ijk_in_unique = np.where(unique_values == img[i, j, k])[0][0]
                    if counts[idx_ijk_in_unique] == counts[min_count_index]:
                        img[i, j, k] = value_with_max_count
                        #print(f'?')


    return img



org_img_path = './d_data/refined_FIB_SEM_high_res_stack/refined_3D_FIB_SEM_sample_0_1_0.tif'


img = tiff.imread(org_img_path)

img[img<50] = 0
img[(img >= 50)&(img < 200)] = 128
img[img >= 200] = 255

pixel_value_Ni = 128
pixel_value_YSZ = 255
pixel_value_pore = 0

img_refined = delete_black_spots(img, pixel_value_pore, pixel_value_YSZ, pixel_value_Ni)


#img_refined = img
tiff.imwrite('./d_data/smoothed_refined_3D_FIB_SEM_sample_0_1_0.tif',
            img_refined.astype(np.uint8), 
            imagej=True, 
            bigtiff=False) 


