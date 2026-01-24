"""
Constants related to dataset
"""
import numpy as np

# List all potential input variables
ALL_VARIABLES=['DBZ',
               'VEL',
               'KDP',
               'RHOHV',
               'ZDR',
               'WIDTH']

# Provides a typical min-max range for each variable (but not exact)
# Used for normalizing in a NN
CHANNEL_MIN_MAX = {
    'DBZ': [-20.,60.],
    'VEL': [-60.,60.],
    'KDP': [-2.,5.],
    'RHOHV': [0.2, 1.04],
    'ZDR': [-1.,8.],
    'WIDTH':[0.,9.]
}














