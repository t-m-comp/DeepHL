# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 05:31:05 2016

@author: ubuntu
"""

datasetrootdir = 'dataset'
resultrootdir = 'result'
modelrootdir = 'model'

# directory name that features of DNN is saved in 
featuredir = 'feature'
# directory name that trajectory is saved in 
trajectorydir = 'trajectory'
# directory name that existing features is saved in 
allfeature = 'allfeature'
# directory name that trajectory (longitude, latitude) is saved in
trajectory_map = 'trajectory_map'

normal = 'normal_preexp'

# mutant = 'dop-3_preexp'
# mutant = 'egl-3_preexp'
mutant = 'egl-21_preexp'

# the data in first "skip" samples is not used
skip = 0
# the number of node in a hidden layer 
num_node = 128
# the number of lstm layer
num_layer = 5
