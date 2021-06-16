"""
The APHRODITE datasets over Monsoon Asia
- V1101 from 1951 to 2007 
- V1101_EXR from 2007-2016
"""

import glob
import xarray as xr
import numpy as np

from tqdm import tqdm

import LocationSel as ls


def collect_APHRO(location):
    """ Downloads data from APHRODITE model"""
    aphro_ds = xr.open_dataset("Data/APHRODITE/aphrodite_indus_1951_2016.nc")
    masked_ds = ls.select_basin(aphro_ds, location, interpolate=True)
    ds = masked_ds.assign_attrs(plot_legend="APHRODITE") # in mm/day   
    return ds


def merge_og_files():
    """ Function to open, crop and merge the APHRODITE data """

    ds_list = []
    extent = ls.basin_extent('indus')

    print('1951-2007')
    for f in tqdm(glob.glob('Data/APHRODITE/APHRO_MA_025deg_V1101.1951-2007.gz/*.nc')):
        ds = xr.open_dataset(f)
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon', 'precip': 'tp'})
        ds_cropped = ds.tp.sel(lon=slice(extent[1], extent[3]), lat=slice(extent[2], extent[0]))
        ds_resampled = (ds_cropped.resample(time="M")).mean()
        ds_resampled['time'] = ds_resampled.time.astype(float)/365/24/60/60/1e9 +1970
        ds_list.append(ds_resampled)

    print('2007-2016')
    for f in tqdm(glob.glob('Data/APHRODITE/APHRO_MA_025deg_V1101_EXR1/*.nc')):
        ds = xr.open_dataset(f)
        ds = ds.rename({'precip': 'tp'})
        ds_cropped = ds.tp.sel(lon=slice(extent[1], extent[3]), lat=slice(extent[2], extent[0]))
        ds_resampled = (ds_cropped.resample(time="M")).mean()
        ds_resampled['time'] = ds_resampled.time.astype(float)/365/24/60/60/1e9 +1970
        ds_list.append(ds_resampled)
    
    ds_merged = xr.merge(ds_list)
    ds_merged.to_netcdf("Data/APHRODITE/aphrodite_indus_1951_2016.nc")
