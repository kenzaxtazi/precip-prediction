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

model_filepath = 'Models/model_2021-01-01/08/21-22-35-56'


def select_coords(lat, lon, dataset):
    """ Interpolate dataset at given coordinates """
    timeseries = dataset.interp(coords={"lon": lon, "lat": lat}, method="nearest")
    timeseries = timeseries.sel(time= slice(1990, 2005))
    return timeseries
        

def select_basin(basin_name, dataset, mask_filepath):  # TODO
    basin_data = dd.apply_mask(dataset, mask_filepath)
    return basin_data

def model_prep(model_filepath, lat, lon):
    """ Prepares model outputs for comparison """

    model = gpm.restore_model(model_filepath)
    xtrain, xval, _, _, _, _ = dp.multivariate_data_prep(coords=[lat,lon])
    
    xtr = np.concatenate((xtrain, xval), axis=0)
    y_gpr, y_std = model.predict_y(xtr)
 
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


def trend_comparison(model_filepath, lat, lon):
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

    tims.timeseries_plot(timeseries, xtr, y_gpr_t, y_std_t)
    dataset_stats(timeseries, xtr, y_gpr_t, y_std_t)
    corr.dataset_correlation(timeseries, xtr, y_gpr_t, y_std_t)


























