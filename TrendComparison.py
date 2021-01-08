# Trend comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cftime
import datetime

import DataDownloader as dd
import DataPreparation as dp
import GPModels as gpm


def collect_data():
    """ Downloads data from other models """
    # ERA5
    mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"
    era5_ds= dd.download_data(mask_filepath, xarray=True)
    era5_utime = era5_ds.time.values / (1e9 * 60 * 60 * 24 * 365)

    # CMIP5 historical
    cmip_59_84_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_Amon_HadCM3_historical_r1i1p1_195912-198411.nc")
    cmip_84_05_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_Amon_HadCM3_historical_r1i1p1_198412-200512.nc")
    cmip_ds = cmip_84_05_ds.merge(cmip_59_84_ds)    
    
    cmip_time = np.array([d.strftime() for d in cmip_ds.time.values])
    cmip_time2 = np.array([datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in cmip_time])
    cmip_utime = np.array([d.timestamp() for d in cmip_time2])/ ( 60 * 60 * 24 * 365)

    # CORDEX 
    cordex_90_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_199001-199012.nc")
    cordex_91_00_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_199101-200012.nc")
    cordex_01_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_200101-201012.nc")
    cordex_02_11_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_201101-201111.nc")
    cordex_90_00_ds = cordex_90_ds.merge(cordex_91_00_ds)
    cordex_01_11_ds= cordex_01_ds.merge(cordex_02_11_ds)
    cordex_ds = cordex_01_11_ds.merge(cordex_90_00_ds)

    cordex_time = cordex_ds.time.values.astype(int)
    cordex_utime = cordex_time / (1e9 * 60 * 60 * 24 * 365)

    return era5_ds, era5_utime, cmip_ds, cmip_utime, cordex_ds, cordex_utime


def select_coords(lat, lon, dataset):
    """ Interpolate dataset at given coordinates """
    try:
        timeseries = dataset.interp(coords={"longitude": lon, "latitude": lat}, method="nearest")
    
    except Exception:
        timeseries = dataset.interp(coords={"lon": lon, "lat": lat}, method="nearest")
    
    return timeseries


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


def plot_comparison(era5_ts, era5_utime, cmip_ts, cmip_utime, cordex_ts, cordex_utime, xtr, y_gpr_t, y_std_t):
    """ Plot of model outputs """
    
    plt.figure()

    plt.plot(cmip_utime + 1970, cmip_ts.pr.values * 60 * 60 * 24, c='b', label='CMIP5 HAD3CM')
    plt.plot(cordex_utime + 1970, cordex_ts.pr.values * 60 * 60 * 24, c='o', label='CORDEX East Asia')
    plt.plot(era5_utime + 1970, era5_ts.tp.values * 1000, c='r', label='ERA5')

    plt.fill_between(xtr[:, 0] + 1979, y_gpr_t[:, 0] - 1.9600 * y_std_t[:, 0], y_gpr_t[:, 0] + 1.9600 * y_std_t[:, 0], 
                        alpha=0.5, color="lightblue", label="95% confidence interval")
    plt.plot(xtr[:, 0] + 1979, y_gpr_t, color="C0", linestyle="-", label="Prediction")

    plt.xlabel('Time')
    plt.ylabel('Precipitation mm/day')
    plt.legend()
    plt.show()


def trend_comparison(model_filepath, lat, lon):
    """ Plots model outputs for given coordinates over time """
    
    era5_ds, era5_utime, cmip_ds, cmip_utime, cordex_ds, cordex_utime = collect_data()
 
    era5_ts = select_coords(lat, lon, era5_ds)
    cmip_ts = select_coords(lat, lon, cmip_ds)
    cordex_ts = select_coords(lat, lon, cordex_ds)

    xtr, y_gpr_t, y_std_t = model_prep(model_filepath, lat, lon)

    plot_comparison(era5_ts, era5_utime, cmip_ts, cmip_utime, cordex_ts, cordex_utime, xtr, y_gpr_t, y_std_t)


























