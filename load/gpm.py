"""
Global Precipitation Measurement (GPM) dataset combines remote sensing data from:
- Tropical Rainfall Measurement Mission (TRMM) between 1997-2015
- Multiple satellite from NASA and JAXA between 2015-present

You'll need to follow the dowload instructions to make this work: 
https://disc.gsfc.nasa.gov/data-access

Delete HDF5 files after making netcdf files (they big).

We are using the Ku beam and Near Surface Precipitation Rate for continutity with TRMM.

"""

import glob
import h5py
import requests

import xarray as xr
import numpy as np
from tqdm import tqdm

import LocationSel as ls

# trmm_filepath = 'Data/GPM/subset_GPM_3PR_06_20210611_090054.txt'
# gpm_filepath = 'Data/GPM'

def hdf5_download(url_filepath):
    """ Dowloads the monthly TRMM/GPM datasets """

    with open(url_filepath) as f:
        for line in tqdm(f):
            url = line[:-1]
            path = 'Data/GPM/'+ url.split('/', -1)[-1]
            r = requests.get(url, allow_redirects=True)
            with open(path, 'wb') as file:
                file.write(r.content)


def to_netcdf():
    """
    Function to open and merge files into one netcdf file.

    HDF5 keys : 
    Level 1 : 'Grids', 'InputAlgorithmVersions', 'InputFileNames', 'InputGenerationDateTimes'
    Level 2 : 'G1' (5.0°x 5.0° grid), 'G2' (0.25°x 0.25° grid)
    Level 3 : 'precipRate', ....
    Level 4 : 'count', 'mean', 'stdev'

    Variables and dimmensions:
        rt (3): rain types (0: stratiform, 1: convective, 2: all)
        chn (5): channels (0: KuNS, 1: KaMS, 2: KaHS, 3: DPRMS, 4: KuMS)
                - Ku (Ku-band Precipitation Radar)
                - Ka (Ka-band Precipitation Radar)
                - DPR (Dual frequency Precipitation Radar)
                - NS (normal scan) = 245km swath
                - MS (matched beam scan) = 125km swath
                - HS (high sensitivity beam scan) = 120km swath
        lnH (1440): high resolution 0.25° grid intervals of longitude from 180°W to 180°E
        ltH (536): high resolution 0.25° grid intervals of latitude from 0.67°S to 0.67°N
    """

    ds_list = []
    extent = ls.basin_extent('indus')
    lon_arr = np.arange(-180, 180, 0.25)
    lat_arr = np.arange(-67, 67, 0.25)

    for file in tqdm(glob.glob('Data/GPM/*.HDF5')):   
        f = h5py.File(file, 'r')
        dset = f['Grids']
        tp_arr = dset['G2']['precipRateNearSurface']['mean'].value
        time = float(file[35:39]) + float(file[39:41])/12.0
        ds = xr.Dataset(data_vars= dict(tp=(["time", "lon", "lat"], [tp_arr[2,0,:,:]])),
                        coords=dict(time=(["time"], [time]), lat=(['lat'], lat_arr), lon=(['lon'], lon_arr)))
        ds_tr = ds.transpose('time', 'lat', 'lon')
        ds_cropped = ds_tr.sel(lon=slice(extent[1], extent[3]), lat=slice(extent[2], 36.0))
        ds_list.append(ds_cropped)
    
    ds_merged = xr.merge(ds_list)
    ds_merged['tp'] = ds_merged['tp']*24  # to mm/day
    ds_merged.to_netcdf("Data/GPM/gpm_2000-2010.nc")


def collect_GPM(location):
    
    gpm_ds = xr.open_dataset("Data/GPM/gpm_2000-2010.nc")
    masked_ds = ls.select_basin(gpm_ds, location, interpolate=True)
    ds = masked_ds.assign_attrs(plot_legend="GPM") # in mm/day   
    return ds


    
    
    

