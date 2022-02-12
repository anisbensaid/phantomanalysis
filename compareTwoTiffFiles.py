#
#Compute the euclidean distance, manhattan distance, and normalized cross-correlation distance between two tif images of the same dimension
#
#
from skimage import io
import numpy as np
from SMF import *
import os
import math
import matplotlib.pyplot as plt
import sys
#file_path = "reconstruction_dataset_org_nopoisson"

#second_file_path = "reconstruction_dataset_org_nopoisson_filtered"
#thrid_file_path = "reconstruction_dataset_org_nopoisson_noise"

file_path = sys.argv[1]

#second_file_path = "rec_res_hotpixel_nopoisson_filtered_SIRT"
#thrid_file_path = "rec_res_hotpixel_nopoisson_SIRT"


second_file_path = sys.argv[2]
thrid_file_path = sys.argv[3]
fig_name = sys.argv[4]

e_dist = []
m_dist = []
ncc_dist = []
#delt_dist = []

def filter(filepath):

    filename = os.path.basename(filepath)
    print(filename)
    prefixname = os.path.splitext(filename)[0]

    input_image = io.imread(filepath).astype(np.int64)
    print(input_image.dtype)
    print(second_file_path+os.sep+filename)
    second_image = io.imread(second_file_path+os.sep+filename)
    third_image = io.imread(thrid_file_path+os.sep+filename)
    ma_dist = np.linalg.norm( input_image - second_image)
    ma_dist2 = np.linalg.norm(input_image - third_image)

    #delta = ma_dist2 - ma_dist
    #print(ma_dist)
    #print(ma_dist2)
    m_dist.append(ma_dist)
    e_dist.append(ma_dist2)
    #delt_dist.append(delta)


for subdir, dirs, files in os.walk(file_path):
    # sort the file names before iterating
    files.sort()
    print(files)
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".tif"):
            filter(filepath)

#print(m_dist)
avg = sum(m_dist) / len(m_dist)
print(avg)
#print(e_dist)
avg2 = sum(e_dist) / len(e_dist)
print(avg2)

#visulize the result

x2 = np.arange(len(m_dist))

np.savetxt("euclideanDistForFiltered_FDK_0.2%.txt", m_dist)
np.savetxt("euclideanDistForFiltered_FDK_grids_blue.txt", e_dist)
#np.savetxt("edForFiltered_FDK_grids_delta.txt", delt_dist)

#plt.plot(xy, dead_precision, "--yo")
plt.plot(x2, m_dist, "--ro")
plt.plot(x2, e_dist, "--bo")
#plt.plot(x2, delt_dist, "--go")
#plt.plot(xy, running_time, "--bo")
#plt.title('Euclidean distance for noisy, filtered minus clean (SIRT) static')
plt.xlabel('Reconstruction slice')
plt.ylabel('Euclidean distance')
plt.legend(['Filtered input', 'Noisy input'], loc='best')
#plt.xticks(x2, xy)
#plt.show()
plt.savefig(fig_name)
