# Trend comparison

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import datetime
import glob
import seaborn as sns
import cftime 
from scipy import stats


import DataDownloader as dd
import DataPreparation as dp
import GPModels as gpm
import Correlation as corr
import Timeseries as tims
import PDF as pdf

model_filepath = 'Models/model_2021-01-01/08/21-22-35-56'


def select_coords(dataset, lat=None, lon=None):
    """ Interpolate dataset at given coordinates """
    timeseries = dataset.interp(coords={"lon": lon, "lat": lat}, method="nearest")
    timeseries = timeseries.sel(time= slice(1990, 2005))
    return timeseries

def select_basin(dataset, location):
    """ Interpolate dataset at given coordinates """  
    mask_filepath = dp.find_mask(location)
    basin = dd.apply_mask(dataset, mask_filepath) 
    basin = basin.sel(time= slice(1990, 2005))  
    return basin

def model_prep(location, data_filepath='Data/model_pred_test.csv', model_filepath='Models/model_2021-01-01/08/21-22-35-56', minyear=1990, maxyear=2005, ):
    """ Prepares model outputs for comparison """

    if os.path.exists(data_filepath):
        model_df = pd.read_csv(data_filepath)

    else:
        model = gpm.restore_model(model_filepath)

        xtr, ytr = dp.areal_model_eval(location, minyear=minyear, maxyear=maxyear)
        y_gpr, y_std = model.predict_y(xtr)
    
        # to mm/day
        y_gpr_t = dp.inverse_log_transform(y_gpr)
        y_std_t = dp.inverse_log_transform(y_std)

        model_df = pd.DataFrame({'time': xtr[:, 0]+1970, 
                                'lat': xtr[:, 1], 
                                'lon': xtr[:, 2], 
                                'tp': y_gpr_t.flatten(),
                                'tp_std': y_std_t.flatten()})

        model_df = model_df.groupby('time').mean()                      
        model_df.to_csv(data_filepath)

    reset_df = model_df.reset_index()
    multi_index_df = reset_df.set_index(["time", "lat", "lon"])

    model_ds = multi_index_df.to_xarray()
    model_ds = model_ds.assign_attrs(plot_legend="Model")
    return model_ds


def dataset_stats(datasets):
    """ Print mean, standard deviations and slope for datasets """

    for ds in datasets:

        tp = ds.tp.values
        if len(tp.shape)>1:
            ds = dp.average_over_coords(ds)
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(ds.time.values, ds.tp.values)
        #print(ds.plot_legend) # TODO
        print('mean = ', np.mean(ds.tp.values),'mm/day')
        print('std = ', np.std(ds.tp.values), 'mm/day')
        print('slope = ', slope, 'mm/day/month')

def single_location_comparison(model_filepath, lat, lon):
    """ Plots model outputs for given coordinates over time """
    
    era5_ds = dd.collect_ERA5()
    cmip_ds = dd.collect_CMIP5()
    cordex_ds = dd.collect_CORDEX()
    cru_ds = dd.collect_CRU()
    #aphro_ds = dd.collect_APHRO()
 
    era5_ts = select_coords(era5_ds, lat, lon)
    cmip_ts = select_coords(cmip_ds, lat, lon)
    cordex_ts = select_coords(cordex_ds, lat, lon)
    cru_ts = select_coords(cru_ds, lat, lon)
    #aphro_ts = select_coords(aphro_ds, lat, lon)

    timeseries = [era5_ts, cmip_ts, cordex_ts, cru_ts] #, aphro_ts]

    xtr, y_gpr_t, y_std_t = model_prep([lat, lon], model_filepath)

    tims.benchmarking_plot(timeseries)
    dataset_stats(timeseries)
    corr.dataset_correlation(timeseries)
    pdf.benchmarking_plot(timeseries)


def basin_comparison(model_filepath, location):
    """ Plots model outputs for given coordinates over time """
    
    era5_ds = dd.collect_ERA5()
    cmip_ds = dd.collect_CMIP5()
    cordex_ds = dd.collect_CORDEX()
    cru_ds = dd.collect_CRU()
    #aphro_ds = dd.collect_APHRO()
 
    era5_bs = select_basin(era5_ds, location)
    cmip_bs = select_basin(cmip_ds, location)
    cordex_bs = select_basin(cordex_ds, location)
    cru_bs = select_basin(cru_ds, location)
    #aphro_bs = select_basin(aphro_ds, location)

    model_bs = model_prep(location, model_filepath)

    basins = [model_bs, era5_bs, cmip_bs, cordex_bs, cru_bs] #, aphro_ts]

    dataset_stats(basins)
    tims.benchmarking_plot(basins)
    corr.dataset_correlation(basins)
    pdf.benchmarking_plot(basins)





























