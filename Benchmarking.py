# Trend comparison

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import cftime 
from scipy import stats

from load import beas_sutlej_gauges, era5, cru, beas_sutlej_wrf, gpm, aphrodite


import DataDownloader as dd
import DataPreparation as dp
import GPModels as gp
import Correlation as corr
import Timeseries as tims
import PDF as pdf
import Maps

from sklearn.metrics import mean_squared_error, r2_score


model_filepath = 'Models/model_2021-01-01/08/21-22-35-56'


def model_prep(location, data_filepath='Data/model_pred_test.csv', model_filepath='Models/model_2021-01-01/08/21-22-35-56', minyear=1990, maxyear=2005, ):
    """ Prepares model outputs for comparison """

    if os.path.exists(data_filepath):
        model_df = pd.read_csv(data_filepath)

    else:
        model = gp.restore_model(model_filepath)

        xtr, ytr = dp.areal_model_eval(location, minyear=minyear, maxyear=maxyear)
        y_gpr, y_std = model.predict_y(xtr)
    
        # to mm/day
        y_gpr_t = dp.inverse_log_transform(y_gpr) * 1000
        y_std_t = dp.inverse_log_transform(y_std) * 1000

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


def dataset_stats(datasets, ref_ds=None):
    """ Print mean, standard deviations and slope for datasets """

    for ds in datasets:

        name = ds.plot_legend
        tp = ds.tp.values
        if len(tp.shape)>1:
            ds = dp.average_over_coords(ds)
        da = ds['tp'].dropna(dim='time')

        slope, intercept, r_value, p_value, std_err = stats.linregress(da.time.values, da.values)
        print(name)
        print('mean = ', np.mean(da.values),'mm/day')
        print('std = ', np.std(da.values), 'mm/day')
        print('slope = ', slope, 'mm/day/year')

        if ref_ds is not None:
            tp_ref = ref_ds.tp.values
            df = pd.DataFrame({'tp_ref': tp_ref, 'tp': tp})
            df = df.dropna()
            r2 = r2_score(df['tp_ref'].values, df['tp'].values)
            rmse = mean_squared_error(df['tp_ref'].values, df['tp'].values, squared=False)
            print('R2 = ', r2)
            print('RMSE = ', rmse)



def single_location_comparison(location=[31.77,76.933]):
    """ Plots model outputs for given coordinates over time """
    
    aphro_ds = aphrodite.collect_APHRO(location, minyear=2000, maxyear=2011)
    cru_ds= cru.collect_CRU(location, minyear=2000, maxyear=2011)
    era5_ds = era5.collect_ERA5(location, minyear=2000, maxyear=2011)
    gpm_ds = gpm.collect_GPM(location,  minyear=2000, maxyear=2011)
    wrf_ds = beas_sutlej_wrf.collect_BC_WRF(location, minyear=2000, maxyear=2011)
    gauge_ds = beas_sutlej_gauges.gauge_download('Banjar', minyear=2000, maxyear=2011)
    
    # cmip_ds = dd.collect_CMIP5()
    # cordex_ds = dd.collect_CORDEX()
    # model_ts = model_prep([lat, lon], data_filepath='single_loc_test.csv', model_filepath=model_filepath)
    
    timeseries = [gauge_ds, era5_ds, gpm_ds, aphro_ds, cru_ds, wrf_ds, ]

    tims.benchmarking_subplots(timeseries, reference_dataset=gauge_ds)
    dataset_stats(timeseries, ref_ds=gauge_ds)
    # corr.dataset_correlation(timeseries)
    pdf.benchmarking_plot(timeseries, kernel_density=False)


def basin_comparison(model_filepath, location):
    """ Plots model outputs for given coordinates over time """
    
    aphro_ds = aphrodite.collect_APHRO(location, minyear=2000, maxyear=2011)
    cru_ds= cru.collect_CRU(location, minyear=2000, maxyear=2011)
    era5_ds = era5.collect_ERA5(location, minyear=2000, maxyear=2011)
    gpm_ds = gpm.collect_GPM(location,  minyear=2000, maxyear=2010.99)
    wrf_ds = beas_sutlej_wrf.collect_BC_WRF(location, minyear=2000, maxyear=2011)
    
    # cmip_ds = dd.collect_CMIP5()
    # cordex_ds = dd.collect_CORDEX()
    # cmip_bs = select_basin(cmip_ds, location)
    # cordex_bs = select_basin(cordex_ds, location)
    # model_bs = model_prep(location, model_filepath)

    basins = [aphro_ds, cru_ds, era5_ds, gpm_ds, wrf_ds]

    dataset_stats(basins)


def multi_location_comparison():
    """ Plots model outputs for given coordinates over time """

    gauge_ds = beas_sutlej_gauges.all_gauge_data(minyear=2000, maxyear=2011)
    locations = [[31.65, 77.34], [31.424, 76.417], [31.80, 77.19], [31.357, 76.878], [31.57, 77.22], 
                 [31.67,77.06], [31.454,77.644], [31.77, 77.31], [31.238,77.108], [31.88, 77.15]]

    aphro_sets = []
    cru_sets = []
    era5_sets = []
    gpm_sets = []
    wrf_sets = []

    for l in locations:
        aphro_ds = aphrodite.collect_APHRO(l, minyear=2000, maxyear=2011)
        aphro_sets.append(aphro_ds.tp)
        
        cru_ds= cru.collect_CRU(l, minyear=2000, maxyear=2011)
        cru_sets.append(cru_ds.tp)
        
        era5_ds = era5.collect_ERA5(l, minyear=2000, maxyear=2011)
        era5_sets.append(era5_ds.tp)
        
        gpm_ds = gpm.collect_GPM(l,  minyear=2000, maxyear=2011)
        gpm_sets.append(gpm_ds.tp)

        wrf_ds = beas_sutlej_wrf.collect_BC_WRF(l, minyear=2000, maxyear=2011)
        wrf_sets.append(wrf_ds.tp)

    # Merge datasets
    aphro_mer_ds = xr.concat(aphro_sets, dim='lon')
    aphro_mer_ds.attrs['plot_legend'] = 'APHRODITE'

    cru_mer_ds = xr.concat(cru_sets, dim='lon')
    cru_mer_ds.attrs['plot_legend'] = 'CRU'

    era5_mer_ds = xr.concat(era5_sets, dim='lon')
    era5_mer_ds.attrs['plot_legend'] = 'ERA5'

    gpm_mer_ds = xr.concat(gpm_sets, dim='lon')
    gpm_mer_ds.attrs['plot_legend'] = 'GPM'

    wrf_mer_ds = xr.concat(wrf_sets, dim='lon')
    wrf_mer_ds.attrs['plot_legend'] = 'WRF_BC'
     
    timeseries = [gauge_ds, era5_mer_ds, gpm_mer_ds, aphro_mer_ds, cru_mer_ds, wrf_mer_ds]
    pdf.mult_gauge_loc_plot((timeseries)





























