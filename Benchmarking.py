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

model_filepath = 'Models/model_2021-01-01/08/21-22-35-56'


def collect_ERA5():
    """ Downloads data from ERA5 """
    mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"
    era5_ds= dd.download_data(mask_filepath, xarray=True) # in m/day
    era5_ds['tp'] *= 1000  # to mm/day
    era5_ds = era5_ds.assign_attrs(plot_legend="ERA5")
    era5_ds['time'] = standardised_time(era5_ds)
    era5_ds = era5_ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    return era5_ds

def collect_CMIP5():
    """ Downloads data from CMIP5 """
    cmip_59_84_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_Amon_HadCM3_historical_r1i1p1_195912-198411.nc")
    cmip_84_05_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_Amon_HadCM3_historical_r1i1p1_198412-200512.nc")
    cmip_ds = cmip_84_05_ds.merge(cmip_59_84_ds)  # in kg/m2/s
    cmip_ds = cmip_ds.assign_attrs(plot_legend="HadCM3 historical")
    cmip_ds = cmip_ds.rename({'pr': 'tp'})
    cmip_ds['tp'] *= 60 * 60 * 24  # to mm/day
    cmip_ds['time'] = standardised_time(cmip_ds)
    return cmip_ds

def collect_CORDEX():
    """ Downloads data from CORDEX East Asia model """
    cordex_90_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_199001-199012.nc")
    cordex_91_00_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_199101-200012.nc")
    cordex_01_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_200101-201012.nc")
    cordex_02_11_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_201101-201111.nc")
    cordex_90_00_ds = cordex_90_ds.merge(cordex_91_00_ds)
    cordex_01_11_ds= cordex_01_ds.merge(cordex_02_11_ds)
    cordex_ds = cordex_01_11_ds.merge(cordex_90_00_ds)  # in kg/m2/s
    
    cordex_ds = cordex_ds.assign_attrs(plot_legend="CORDEX EA - MOHC-HadRM3P historical")
    cordex_ds = cordex_ds.rename_vars({'pr': 'tp'})
    cordex_ds['tp'] *= 60 * 60 * 24   # to mm/day
    cordex_ds['time'] = standardised_time(cordex_ds)

    return cordex_ds

def collect_APHRO():
    """ Downloads data from APHRODITE model"""
    aphro_ds = xr.merge([xr.open_dataset(f) for f in glob.glob('/Users/kenzatazi/Downloads/APHRO_MA_025deg_V1101.1951-2007.gz/*')])
    aphro_ds = aphro_ds.assign_attrs(plot_legend="APHRODITE") # in mm/day   
    aphro_ds = aphro_ds.rename_vars({'precip': 'tp'})
    aphro_ds['time'] = standardised_time(aphro_ds)
    return aphro_ds

def collect_CRU():
    """ Downloads data from CRU model"""
    cru_ds = xr.open_dataset("/Users/kenzatazi/Downloads/cru_ts4.04.1901.2019.pre.dat.nc")
    cru_ds = cru_ds.assign_attrs(plot_legend="CRU") # in mm/month
    cru_ds = cru_ds.rename_vars({'pre': 'tp'})
    cru_ds['tp'] /= 30.437  #TODO apply proper function to get mm/day
    cru_ds['time'] = standardised_time(cru_ds)
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
    timeseries = dataset.interp(coords={"lon": lon, "lat": lat}, method="nearest")
    timeseries = timeseries.sel(time= slice(1990, 2005))
    return timeseries
        

def select_basin(basin_name, dataset):  # TODO
    basin_data = dd.apply_mask(dataset)
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


def timeseries_plot(timeseries, xtr, y_gpr_t, y_std_t):
    """ Plot timeseries of model outputs """
    plt.figure()
    for ts in timeseries:
        plt.plot(ts.time.values, ts.tp.values, label=ts.plot_legend)
    plt.fill_between(xtr[:, 0] + 1979, y_gpr_t[:, 0] - 1.9600 * y_std_t[:, 0], y_gpr_t[:, 0] + 1.9600 * y_std_t[:, 0], 
                        alpha=0.5, color="lightblue") #label="95% confidence interval")
    plt.plot(xtr[:, 0] + 1979, y_gpr_t, linestyle="-", label="Prediction")
    plt.xlabel('Time')
    plt.ylabel('Precipitation mm/day')
    plt.legend()
    plt.show()



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

    timeseries_plot(timeseries, xtr, y_gpr_t, y_std_t)
    dataset_stats(timeseries, xtr, y_gpr_t, y_std_t)
    corr.dataset_correlation(timeseries)


























