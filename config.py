import numpy as np

WORKING_DIR = "."

block_height = -45

default_port="com21"  # for Windows

#  3x3 affine matrix for pixel -> robot (X,Y)
M = np.array([
     [1.38650232e-03 ,-4.57354952e-01,  4.3830889e+02],
     [-4.85095919e-01  ,2.06416755e-03,  1.53159675e+02]
], dtype=np.float64)
                 
z_above = 50          # safe travel height (e.g. 100)
z_table = -59           # Z at table contact
block_height_mm = 10   # block physical thickness
block_length_mm = 18   # block physical length
stack_delta_mm = 8    # extra height when stacking (to avoid collision)
side_offset_mm = 10    # extra XY gap when placing beside

capture_wait_time = 8
camera_index = 2