# Selective median filter
# Input: 1. Path for tif image file or folder
#        2. Kernel size (Default 3)
#        3. Threshold   (Default 50)

# Output: 1. Filtered image(s) will be saved in the output folder
#         2. Dead pixels mask
#         3. Hot pixels mask

# Function:
#         Performs a selected median filter on the image
#         detect the hot and dead pixels of an input image
#         Detection is written to a hot and dead mask with 1 indicates there is an outlier
#         Along with the mask, there is a confidence map indicating how confident the detection is

#Author: Yin Huang
#Date: Sep 15, 2020
import numpy as np
from matplotlib import pyplot as pt

def hotdeaddetection(arr1, arr2, res, display_min, display_max, threshold, hotmask, deadmask, confidence_map_dead, confidence_map_hot):
    normalization_confidence_map = (display_max - display_min)
    # Sequentially compare the pixel values
    for ix, iy in np.ndindex(arr1.shape):
        # Noisy pixel
        val1 = arr1[ix, iy]
        # Median pixel
        val2 = arr2[ix, iy]
        # difference
        delta = abs(int(val1) - int(val2))

        # When delta is larger than threshold, an outlier is detected
        if delta >= threshold:
            # case dead pixel
            if val1 == display_min:
                # Update res using median value
                res[ix, iy] = val2
                # Mask the coordinate as true for dead mask
                deadmask[ix, iy] = True
                # Calculate the confidence value
                confidence_map_dead[ix, iy] \
                    = delta / normalization_confidence_map

            # case hot pixel
            else:
                # Update res using median value
                res[ix, iy] = val2
                # Mask the coordinate as true for hot mask
                hotmask[ix, iy] = True
                # Calculate the confidence value
                confidence_map_hot[ix, iy] \
                    = delta / normalization_confidence_map


def show(arr, title):
    pt.figure()
    pt.title(title)
    pt.imshow(arr, cmap='gray')
    pt.show()


def hotdeaddetection_v(arr1, arr2, res, display_min, display_max, threshold, hotmask, deadmask, confidence_map_dead, confidence_map_hot):
    normalization_confidence_map = (display_max - display_min)
    # Sequentially compare the pixel values
    for ix, iy in np.ndindex(arr1.shape):
        # Noisy pixel
        val1 = arr1[ix, iy]
        # Median pixel
        val2 = arr2[ix, iy]
        # difference
        delta = abs(int(val1) - int(val2))

        # When delta is larger than threshold, an outlier is detected
        if delta >= threshold:
            # case dead pixel
            if val1 == display_min:
                # Update res using median value
                res[ix, iy] = val2
                # Mask the coordinate as true for dead mask
                deadmask[ix, iy] = True
                # Calculate the confidence value
                confidence_map_dead[ix, iy] \
                    = val1

            # case hot pixel
            else:
                # Update res using median value
                res[ix, iy] = val2
                # Mask the coordinate as true for hot mask
                hotmask[ix, iy] = True
                # Calculate the confidence value
                confidence_map_hot[ix, iy] \
                    = val1
