# Trend comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cftime
import datetime
import glob
import seaborn as sns

from scipy import stats

import DataDownloader as dd
import DataPreparation as dp
import GPModels as gpm
import Correlation as corr
import Timeseries as tims
import PDF as pdf

model_filepath = 'Models/model_2021-01-01/08/21-22-35-56'

# Masks
uib_mask = 'Data/Masks/ERA5_Upper_Indus_mask.nc'
beas_mask = 'Data/Masks/Beas_basin_overlap.nc'
sutlej_mask = 'Data/Masks/Sutlej_basin_overlap.nc'
gilgit_mask = 'Data/Masks/Gilgit_mask.nc'
khyber_mask = 'Data/Masks/Khyber_mask.nc'
ngari_mask = 'Data/Masks/Khyber_mask.nc'

def select_coords(dataset, lat=None, lon=None):
    """ Interpolate dataset at given coordinates """
    timeseries = dataset.interp(coords={"lon": lon, "lat": lat}, method="nearest")
    timeseries = timeseries.sel(time= slice(1990, 2005))

def select_basin(dataset, mask_filepath):
    """ Interpolate dataset at given coordinates """  
    basin = dd.apply_mask(dataset, mask_filepath) 
    basin = timeseries.sel(time= slice(1990, 2005))  
    return basin


def model_prep(model_filepath, lat=None, lon=None, mask_filepath=None):
    """ Prepares model outputs for comparison """

    model = gpm.restore_model(model_filepath)

    if lat != None:
        xtrain, xval, _, _, _, _ = dp.areal_model_eval(coords=[lat,lon])
    
    if mask_filepath != None:
        xtrain, xval, _, _, _, _ = dp.areal_model(length=3000, mask=None)
    
    xtr = np.concatenate((xtrain, xval), axis=0)
    y_gpr, y_std = model.predict_y(xtr[132:312, 0])
 
    # to mm/day
    y_gpr_t = dp.inverse_log_transform(y_gpr) * 1000
    y_std_t = dp.inverse_log_transform(y_std) * 1000

    return xtr, y_gpr_t, y_std_t


def dataset_stats(datasets, xtr, y_gpr_t, y_std_t):
    """ Print mean, standard deviations and slope for datasets """

    for ds in datasets:
        slope, intercept, r_value, p_value, std_err = stats.linregress(ds.time.values, ds.tp.values)
        print(ds.plot_legend)
        print('mean = ', np.mean(ds.tp.values),'mm/day')
        print('std = ', np.std(ds.tp.values), 'mm/day')
        print('slope = ', slope, 'mm/day/month')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(xtr[131:313, 0], y_gpr_t[131:313, 0]) # btw 1990 and 2005
    print("model")
    print('mean = ', np.mean(y_gpr_t),'mm/day')
    print('std = ', np.std(y_gpr_t), 'mm/day')
    print('slope = ', slope, 'mm/day/month')


def single_location_comparison(model_filepath, lat, lon):
    """ Plots model outputs for given coordinates over time """
    
    era5_ds = dd.collect_ERA5()
    cmip_ds = dd.collect_CMIP5()
    cordex_ds = dd.collect_CORDEX()
    cru_ds = dd.collect_CRU()
    #aphro_ds = dd.collect_APHRO()
 
    era5_ts = select_coords(lat, lon, era5_ds)
    cmip_ts = select_coords(lat, lon, cmip_ds)
    cordex_ts = select_coords(lat, lon, cordex_ds)
    cru_ts = select_coords(lat, lon, cru_ds)
    #aphro_ts = select_coords(lat, lon, aphro_ds)

    timeseries = [era5_ts, cmip_ts, cordex_ts, cru_ts] #, aphro_ts]

    xtr, y_gpr_t, y_std_t = model_prep(model_filepath, lat, lon)

    tims.benchmarking_plot(timeseries, xtr, y_gpr_t, y_std_t)
    dataset_stats(timeseries, xtr, y_gpr_t, y_std_t)
    corr.dataset_correlation(timeseries, y_gpr_t)
    pdf.benchmarking_plot(timeseries, y_gpr_t)


def basin_comparison(model_filepath, lat, lon):
    """ Plots model outputs for given coordinates over time """
    
    era5_ds = dd.collect_ERA5()
    cmip_ds = dd.collect_CMIP5()
    cordex_ds = dd.collect_CORDEX()
    cru_ds = dd.collect_CRU()
    #aphro_ds = dd.collect_APHRO()
 
    era5_ts = select_basin(era5_ds, mask_filepath)
    cmip_ts = select_basin(cmip_ds, mask_filepath)
    cordex_ts = select_basin(cordex_ds, mask_filepath)
    cru_ts = select_basin(cru_ds, mask_filepath)
    #aphro_ts = select_basin(aphro_ds, mask_filepath)

    timeseries = [era5_ts, cmip_ts, cordex_ts, cru_ts] #, aphro_ts]

    xtr, y_gpr_t, y_std_t = model_prep(model_filepath, lat, lon)

    tims.benchmarking_plot(timeseries, xtr, y_gpr_t, y_std_t, basin=True)
    dataset_stats(timeseries, xtr, y_gpr_t, y_std_t, basin=True)
    corr.dataset_correlation(timeseries, y_gpr_t, basin=True)
    pdf.benchmarking_plot(timeseries, y_gpr_t, basin=True)


























