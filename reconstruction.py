from __future__ import division
 
import numpy as np
from os import mkdir
from os.path import join, isdir
from imageio import imread, imwrite
import sys 
import astra

# python reconstruction.py input algorithm

# Configuration.
distance_source_origin = 150  # [mm]
distance_origin_detector = 50  # [mm]
detector_pixel_size = 1.05  # [mm]
#detector_rows = 200  # Vertical size of detector [pixels].
#detector_cols = 200  # Horizontal size of detector [pixels].
detector_rows = 1024
detector_cols = 1024
detector_z = 256
#detector_rows = 3520
#detector_cols = 4656


num_of_projections = 60
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)
#input_dir = 'input_outside_nopoisson'
#input_dir= 'output_hotpixel_various_horizontal_streaks_mid_filtered'
#output_dir = 'reconstruction_output_hotpixel_various_horizontal_streaks_mid_filtered_FDK'
#input_dir = 'phantom_256_noisy_static_filtered_1000'
#output_dir = 'rec_res_phantom_256_noisy_static_filtered_fdk'
#input_dir = 'phantom_256_parallel_attenuated'
#output_dir = 'phantom_256_parallel_attenuated_cgls'

input_dir = sys.argv[1]
algorithm_used = sys.argv[2]
output_dir = input_dir+'_'+ algorithm_used;

# Load projections.
projections = np.zeros((detector_rows, num_of_projections, detector_cols))
for i in range(num_of_projections):
    im = imread(join(input_dir, 'proj%04d.tif' % i)).astype(float)
    im /= 65535
    projections[:, i, :] = im
 
# Copy projection images into ASTRA Toolbox.
#proj_geom = \
#  astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
#                         (distance_source_origin + distance_origin_detector) /
#                         detector_pixel_size, 0)
proj_geom = \
   astra.create_proj_geom('parallel3d', 1, 1, detector_rows, detector_cols, angles)

projections_id = astra.data3d.create('-sino', proj_geom, projections)
 
# Create reconstruction.
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_rows,
                                          detector_z)
reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
#alg_cfg = astra.astra_dict('FDK_CUDA')
alg_cfg = astra.astra_dict(algorithm_used.upper()+'3D_CUDA')
alg_cfg['ProjectionDataId'] = projections_id
alg_cfg['ReconstructionDataId'] = reconstruction_id
algorithm_id = astra.algorithm.create(alg_cfg)
astra.algorithm.run(algorithm_id)
reconstruction = astra.data3d.get(reconstruction_id)
 
# Limit and scale reconstruction.
reconstruction[reconstruction < 0] = 0
reconstruction /= np.max(reconstruction)
reconstruction = np.round(reconstruction * 65535).astype(np.uint16)
 
# Save reconstruction.
if not isdir(output_dir):
    mkdir(output_dir)
for i in range(detector_z):
    im = reconstruction[i, :, :]
    im = np.flipud(im)
    imwrite(join(output_dir, 'reco%04d.tif' % i), im)
 
# Cleanup.
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)



