# Trend comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cftime
import datetime
import glob

import DataDownloader as dd
import DataPreparation as dp
import GPModels as gpm


def collect_ERA5():
    """ Downloads data from ERA5 """
    mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"
    era5_ds= dd.download_data(mask_filepath, xarray=True)
    era5_ds = era5_ds.assign_attrs(plot_legend="ERA5")
    return era5_ds

def collect_CMIP5():
    """ Downloads data from CMIP5 """
    cmip_59_84_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_Amon_HadCM3_historical_r1i1p1_195912-198411.nc")
    cmip_84_05_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_Amon_HadCM3_historical_r1i1p1_198412-200512.nc")
    cmip_ds = cmip_84_05_ds.merge(cmip_59_84_ds)
    cmip_ds = cmip_ds.assign_attrs(plot_legend="HadCM3 historical")
    cmip_ds = cmip_ds.rename_vars({'pr': 'tp'})
    return cmip_ds

def collect_CORDEX():
    """ Downloads data from CORDEX East Asia model """
    cordex_90_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_199001-199012.nc")
    cordex_91_00_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_199101-200012.nc")
    cordex_01_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_200101-201012.nc")
    cordex_02_11_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_201101-201111.nc")
    cordex_90_00_ds = cordex_90_ds.merge(cordex_91_00_ds)
    cordex_01_11_ds= cordex_01_ds.merge(cordex_02_11_ds)
    cordex_ds = cordex_01_11_ds.merge(cordex_90_00_ds)
    cordex_ds = cordex_ds.assign_attrs(plot_legend="CORDEX EA - MOHC-HadRM3P historical")   
    cordex_ds = cordex_ds.rename_vars({'pr': 'tp'})
    return cordex_ds

def collect_APHRO():
    """ Downloads data from APHRODITE model"""
    aphro_ds = xr.merge([xr.open_dataset(f) for f in glob.glob('/Users/kenzatazi/Downloads/APHRO_MA_025deg_V1101.1951-2007.gz/*')])
    aphro_ds = aphro_ds.assign_attrs(plot_legend="APHRODITE")   
    aphro_ds = aphro_ds.rename_vars({'pr': 'tp'})
    return aphro_ds

def collect_CRU():
    """ Downloads data from CRU model"""
    cru_ds = xr.open_dataset("/Users/kenzatazi/Downloads/cru_ts4.04.1901.2019.pre.dat.nc")
    cru_ds = cru_ds.assign_attrs(plot_legend="CRU")
    cru_ds = cru_ds.rename_vars({'pre': 'tp'})
    return cru_ds

def standardised_time(dataset):
    """ Returns array of standardised times to plot """
    try:
        utime = dataset.time.values.astype(int)/(1e9 * 60 * 60 * 24 * 365)
    except Exception:
        time = np.array([d.strftime() for d in dataset.time.values])
        time2 = np.array([datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in time])
        utime = np.array([d.timestamp() for d in time2])/ ( 60 * 60 * 24 * 365)
    return(utime + 1970)


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
    y_gpr, y_std = model.predict_y(xval)
 
    # to mm/day
    y_gpr_t = dp.inverse_log_transform(y_gpr) * 1000
    y_std_t = dp.inverse_log_transform(y_std) * 1000

    return xtr, y_gpr_t, y_std_t


def timeseries_comparison(timeseries, xtr, y_gpr_t, y_std_t):
    """ Plot timeseries of model outputs """
    
    plt.figure()

    for ts in timeseries:
        time = standardised_time(ts)
        plt.plot(time, ts.pr.values, label=ts.plot_legend)

    plt.fill_between(xtr[:, 0] + 1979, y_gpr_t[:, 0] - 1.9600 * y_std_t[:, 0], y_gpr_t[:, 0] + 1.9600 * y_std_t[:, 0], 
                        alpha=0.5, color="lightblue", label="95% confidence interval")
    plt.plot(xtr[:, 0] + 1979, y_gpr_t, color="C0", linestyle="-", label="Prediction")

    plt.xlabel('Time')
    plt.ylabel('Precipitation mm/day')
    plt.legend()
    plt.show()


def pdf_comparison(timeseries, xtr, y_gpr_t, y_std_t):
    """ Plot probability distribution of model outputs """
    return 1


def rolling_timseries_comparison(timeseries, xtr, y_gpr_t, y_std_t):
    """ Plot rolling avereaged timeseries of model outputs """
    
    return 1


def trend_comparison(model_filepath, lat, lon):
    """ Plots model outputs for given coordinates over time """
    
    era5_ds = collect_ERA5()
    cmip_ds = collect_CMIP5()
    cordex_ds = collect_CORDEX()
    cru_ds = collect_CRU()
    #aphro_ds = collect_APHRO()
 
    era5_ts = select_coords(lat, lon, era5_ds)
    cmip_ts = select_coords(lat, lon, cmip_ds)
    cordex_ts = select_coords(lat, lon, cordex_ds)
    cru_ts = select_coords(lat, lon, cru_ds)
    #aphro_ts = select_coords(lat, lon, aphro_ds)

    timeseries = [era5_ts, cmip_ts, cordex_ts, cru_ts] #, aphro_ts]

    xtr, y_gpr_t, y_std_t = model_prep(model_filepath, lat, lon)

    timeseries_comparison(timeseries, xtr, y_gpr_t, y_std_t)

























