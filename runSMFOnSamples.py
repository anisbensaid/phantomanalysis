'''
Run SMF on the samples_every_6_degrees
Save the results as well as the detected hot pixels in samples_every_6_degree_noise and in samples_every_6_degree_filtered.
'''
from skimage import io
import numpy as np
from SMF import *
from scipy import ndimage
import os
from sklearn.metrics import confusion_matrix as cm
import sys

hot_pixel_val = 65504   # hot pixel value from NCNR is 65504
#noise_level = 0.002    # noise level is 0.22%

kernel_size = 5
threshold = 5000

#file_path = "samples_every_6_degrees_test"
file_path = sys.argv[1]
output_path = file_path+"_filtered_"+str(threshold)  # save the filtered results in this folder
output_noise = file_path+"_noise_"+str(threshold)    # save the detected outliers in this folder

if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(output_noise):
    os.makedirs(output_noise)

height = int(sys.argv[2])
width = int(sys.argv[3])

def filter(filepath):
    # load the test image
    filename = os.path.basename(filepath)
    prefixname = os.path.splitext(filename)[0]

    input_image = io.imread(filepath)

    #input_clean = np.array(input_image)
    # show(input_clean, "Original image")
    #input_image[hot_gt_mask] = hot_pixel_val  # 65504 is found in most NCNR hot pixels
    # show(input_image, "Original image with noise")
    # save the noise image
    #outputname_noise = output_path + os.sep + prefixname + '.tif'
    #io.imsave(outputname_noise, input_image)

    median = ndimage.median_filter(input_image, kernel_size)
    res = np.array(input_image)
    hot_detected_mask = np.zeros((height, width), dtype=bool)
    dead_detected_mask = np.zeros((height, width), dtype=bool)

    confidence_map_hot = np.zeros((height, width), dtype=np.uint16)
    confidence_map_dead = np.zeros((height, width), dtype=np.uint16)

    # modify the hot_detected_mask and dead_detected_mask to save the detected pixel values instead of the confidence value
    hotdeaddetection_v(input_image, median, res, 0, hot_pixel_val, threshold, hot_detected_mask,
                     dead_detected_mask,
                     confidence_map_dead, confidence_map_hot)


    io.imsave(output_path+os.sep+prefixname+'.tif', res)
    io.imsave(output_noise+os.sep+prefixname+'.tif', confidence_map_hot)
    #save confidence_map
    #np.savetxt("testSMFOnPhantomData_output/confidence_map_hot_256.csv", confidence_map_hot, delimiter=",")

for subdir, dirs, files in os.walk(file_path):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".tif"):
            filter(filepath)
