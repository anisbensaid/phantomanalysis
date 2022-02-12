'''
Run statistical analysis on the detected noise results
Input::   samples_every_6_degrees
Function:
        Save the files in this directory into an array, each array is a 2D matrix
        1. Check if all element at pos [x,y]  all equal to 0, if yes, normal pixel
        2. Check if any element is 0, if yes, non_constant_hot_pixel
        3. Check if the elements are next to each other, if yes, constant_hot_pixel
        4. otherwise, abnormal_elements
'''
from skimage import io
import numpy as np
from SMF import *
from scipy import ndimage
import os
from sklearn.metrics import confusion_matrix as cm

file_path = "samples_every_6_degrees_noise_5000"
#file_path = "real_data_r"
all_files = []

rows = 3520
cols = 4656

threshold = 10000

def loadfiles(filepath):
    # load the images into files
    filename = os.path.basename(filepath)
    prefixname = os.path.splitext(filename)[0]

    input_image = io.imread(filepath)
    all_files.append(input_image)

for subdir, dirs, files in os.walk(file_path):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".tif"):
            loadfiles(filepath)

# save the counts for each categories
cnt_normal = 0
cnt_nonconstant_hot_pixel = 0
cnt_constant_hot_pixel = 0
cnt_abnormal_hot_pixel = 0


normal_mask = np.zeros((rows, cols), dtype=bool)
nonconstant_mask = np.zeros((rows, cols), dtype=bool)
constant_mask = np.zeros((rows, cols), dtype=bool)
ab_mask = np.zeros((rows, cols), dtype=bool)

normal_arr = np.zeros((rows, cols))
noncostant_arr = np.zeros((rows, cols))
constant_arr = np.zeros((rows, cols))
ab_arr = np.zeros((rows, cols))


# save the results for each category
normal_rows = []
normal_cols = []

nonconstant_rows = []
nonconstant_cols = []

constant_rows = []
constant_cols = []

ab_rows = []
ab_cols = []

for x in range(0, rows):
    for y in range(0, cols):
        min_val = 65536
        max_val = 0
        see_zero_so_far = False
        see_value_so_far = False
        part_zeros = False

        for file in all_files:
            if file[x, y] < min_val:
                min_val = file[x, y]
            if file[x, y] > max_val:
                max_val = file[x, y]
            if ( file[x,y] != 0 ):
            # non - 0 found
                see_value_so_far = True
                if( see_zero_so_far ):
                    part_zeros = True
                    break
            else:
            # 0 found
                see_zero_so_far = True
                if( see_value_so_far ):
                    part_zeros = True
                    break

        if( part_zeros ):
            # save part_zeros case
            nonconstant_rows.append(x)
            nonconstant_cols.append(y)
            cnt_nonconstant_hot_pixel = cnt_nonconstant_hot_pixel + 1
        elif see_zero_so_far:
            # save normal case
            normal_rows.append(x)
            normal_cols.append(y)
            cnt_normal = cnt_normal + 1
        else:
            if ( abs(max_val - min_val) <= threshold ):
                # save constant case
                constant_rows.append(x)
                constant_cols.append(y)
                cnt_constant_hot_pixel = cnt_constant_hot_pixel + 1
            else:
                # save non-constant case
                ab_rows.append(x)
                ab_cols.append(y)
                cnt_abnormal_hot_pixel = cnt_abnormal_hot_pixel + 1


normal_mask = np.zeros((rows, cols), dtype=bool)
nonconstant_mask = np.zeros((rows, cols), dtype=bool)
constant_mask = np.zeros((rows, cols), dtype=bool)
ab_mask = np.zeros((rows, cols), dtype=bool)

normal_arr = np.zeros((rows, cols))
noncostant_arr = np.zeros((rows, cols))
constant_arr = np.zeros((rows, cols))
ab_arr = np.zeros((rows, cols))

normal_mask[normal_rows, normal_cols] = True
nonconstant_mask[nonconstant_rows, nonconstant_cols] = True
constant_mask[constant_rows, constant_cols] = True
ab_mask[ab_rows, ab_cols] = True


normal_arr[normal_mask] = 1
normal_tif = np.array(normal_arr * 65535).astype(np.uint16)
print("normal_tif nonzeros: " + str(np.count_nonzero(normal_tif)))
io.imsave("normal.tif", normal_tif)

noncostant_arr[nonconstant_mask] = 1
nonconstant_tif = np.array(noncostant_arr * 65535).astype(np.uint16)
print("nonconstant_tif nonzeros: " + str(np.count_nonzero(nonconstant_tif)))
io.imsave("nonconstant.tif", nonconstant_tif)

constant_arr[constant_mask] = 1
constant_tif = np.array(constant_arr * 65535).astype(np.uint16)
print("constant_tif nonzeros: " + str(np.count_nonzero(constant_tif)))
io.imsave("constant.tif", constant_tif)

ab_arr[ab_mask] = 1
ab_tif = np.array(ab_arr * 65535).astype(np.uint16)
print("ab_tif nonzeros: " + str(np.count_nonzero(ab_tif)))
io.imsave("ab.tif", ab_tif)

total_cnt = 16389120

print("cnt_normal: " + str(cnt_normal) + " perc: " + str(cnt_normal/total_cnt))
print("cnt_nonconstant_hot_pixel: " + str(cnt_nonconstant_hot_pixel)  + " perc: " + str(cnt_nonconstant_hot_pixel/total_cnt) )
print("cnt_constant_hot_pixel: " + str(cnt_constant_hot_pixel)+ " perc: " + str(cnt_constant_hot_pixel/total_cnt))
print("cnt_abnormal_hot_pixel: "+ str(cnt_abnormal_hot_pixel)+ " perc: " + str(cnt_abnormal_hot_pixel/total_cnt))
