'''
add real noise to the phantom object
Real noise image: constant.tif
Crop start point location: (x, y)

attenuation transform of the phantom object

iterate the phantom object, if the cropped noisy image is not 0, add either static hot pixel or dynamic hot pixel to the location

'''
from skimage import io
import numpy as np
from SMF import *
import os
import random


noisy_path = "constant.tif"

noisy_image = io.imread(noisy_path)

#dimension of the phantom object
phantom_obj_dimx = 1024
phantom_obj_dimy = 1024

#location of the cropped image (start point)
start_point_x = 1500
start_point_y = 2000

camera_height = 3520
camera_width = 4656

cropped_image = np.full((phantom_obj_dimx, phantom_obj_dimy), 0)

#copy the pixel value from the noisy path to the cropped_image
for x in range(phantom_obj_dimx):
    for y in range(phantom_obj_dimy):
        if noisy_image[x + start_point_x, y+start_point_y] !=0 :
            cropped_image[x, y] = noisy_image[x + start_point_x, y+start_point_y]

#input file folder for phantom objects
file_path = "test_generation_256_parallel_60"

#below two types of noisy ones
output_path_static = "phantom_256_parallel_60_static"
output_path_dynamic = "phantom_256_parallel_60_dynamic"

#below is the clean one
output_path_attenuated = "phantom_256_parallel_60_attenuated"

#below is only adding one noise in the mid
output_path_mid = "phantom_256_parallel_60_mid"

darkCounts = 100
dark_field = np.full((phantom_obj_dimx, phantom_obj_dimy), darkCounts, dtype=np.float16)

flatCounts = 1000

hot_pixel_max = 65535
hot_pixel_min = 45535

#fixed location noise
coor_x = 512
coor_y = 512


def addRealNoise(filepath):
    filename = os.path.basename(filepath)
    prefixname = os.path.splitext(filename)[0]
    input_image = io.imread(filepath)
    # convert to float type
    input_float = (input_image / 65535).astype(np.float32)

    # I = I0*exp(-OD) + darkfield;

    input_float = dark_field + flatCounts * np.exp(-input_float)

    input_float /= np.max(input_float)

    # input_clean[hot_gt_mask] = hot_pixel_val
    intput_float_noisy = np.array(input_float)
    intput_float_noisy = np.round(intput_float_noisy * 65535).astype(np.uint16)

    io.imsave(output_path_attenuated + os.sep + prefixname + '.tif', intput_float_noisy)


    ##now iterate the cropped_image and if it is not 0, add either dynamic or static hot pixel
    static_noisy = np.array(intput_float_noisy)
    dynamic_noisy = np.array(intput_float_noisy)

    #fixed location noise
    mid_noisy = np.array(intput_float_noisy)
    mid_noisy[coor_x, coor_y] = hot_pixel_max


    for i in range(phantom_obj_dimx):
        for j in range(phantom_obj_dimy):
            if cropped_image[i, j] != 0:
                static_noisy[i, j] = hot_pixel_max
                dynamic_noisy[i, j] = random.randint(hot_pixel_min, hot_pixel_max)

    io.imsave(output_path_static + os.sep + prefixname + '.tif', static_noisy)
    io.imsave(output_path_dynamic + os.sep + prefixname + '.tif', dynamic_noisy)

    io.imsave(output_path_mid + os.sep + prefixname + '.tif', mid_noisy)


for subdir, dirs, files in os.walk(file_path):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".tif"):
            addRealNoise(filepath)
