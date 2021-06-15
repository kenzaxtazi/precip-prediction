"""
Datasets should aim to all have similar formats so they can be easily used and compared.

- lon for longitude in °E (float)
- lat for latitude in °N (float)
- time for time in years (float)
- tp for total precipitation in mm/day/month (float)

Datasets should be returned as xarray Datasets and saved as netcdf files when possible.
"""


# Dataset class

'''
class Dataset(dataset, timerange=None, location=None):

    def __init__(self):
        
        # load dataset from name given
        # mask or interp to location
        # slice to time range
'''