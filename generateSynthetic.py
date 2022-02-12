from __future__ import division
 
import numpy as np
from os import mkdir
from os.path import join, isdir
from imageio import get_writer
 
import astra
 
# Configuration.
distance_source_origin = 150  # [mm]
distance_origin_detector = 50  # [mm]
detector_pixel_size = 1.05  # [mm]
#detector_rows = 200  # Vertical size of detector [pixels].
#detector_cols = 200  # Horizontal size of detector [pixels].

detector_rows = 1024
detector_cols = 1024
detector_z = 256
#detector_rows = 1200
#detector_cols = 1600
#detector_z = 1280

num_of_projections = 60
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)
output_dir = 'test_generation_256_parallel_60'
 
# Create phantom.
#create vol_geom: y, x z
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_rows,
                                          detector_z)
#create with an array it is z y x
phantom = np.zeros((detector_z, detector_cols, detector_rows))

'''
hb = 110  # Height of beam [pixels].
wb = 40   # Width of beam [pixels].
hc = 100  # Height of cavity in beam [pixels].
wc = 30   # Width of cavity in beam [pixels].
'''
hb = 240
wb = 500
hc = 220
wc = 400

phantom[detector_z // 2 - hb // 2 : detector_z // 2 + hb // 2,
        detector_cols // 2 - wb // 2 : detector_cols // 2 + wb // 2,
        detector_rows // 2 - wb // 2 : detector_rows // 2 + wb // 2] = 1
phantom[detector_z // 2 - hc // 2 : detector_z // 2 + hc // 2,
        detector_cols // 2 - wc // 2 : detector_cols // 2 + wc // 2,
        detector_rows // 2 - wc // 2 : detector_rows // 2 + wc // 2] = 0
phantom[detector_z // 2 - 15 :       detector_z // 2 + 15,
        detector_cols // 2 + wc // 2 : detector_cols // 2 + wb // 2,
        detector_rows // 2 - 15 :       detector_rows // 2 + 15] = 0
phantom_id = astra.data3d.create('-vol', vol_geom, data=phantom)
 
# Create projections. With increasing angles, the projection are such that the
# object is rotated clockwise. Slice zero is at the top of the object. The
# projection from angle zero looks upwards from the bottom of the slice.
#proj_geom = astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles, (distance_source_origin + distance_origin_detector) / detector_pixel_size, 0)
proj_geom = astra.create_proj_geom('parallel3d', 1, 1, detector_rows, detector_cols, angles)
projections_id, projections = astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)
projections /= np.max(projections)
 
# Apply Poisson noise.
#projections = np.random.poisson(projections * 10000) / 10000
#projections[projections > 1.1] = 1.1
#projections /= 1.1
 
# Save projections.
if not isdir(output_dir):
    mkdir(output_dir)
projections = np.round(projections * 65535).astype(np.uint16)
for i in range(num_of_projections):
    projection = projections[:, i, :]
    with get_writer(join(output_dir, 'proj%04d.tif' %i)) as writer:
        writer.append_data(projection, {'compress': 9})
 
# Cleanup.
astra.data3d.delete(projections_id)
astra.data3d.delete(phantom_id)
