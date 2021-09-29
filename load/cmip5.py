import xarray as xr
import numpy as np
import datetime


def collect_CMIP5():
    """ Downloads data from CMIP5 """
    cmip_59_84_ds = xr.open_dataset(
        "_Data/cmip5/pr_Amon_HadCM3_historical_r1i1p1_195912-198411.nc")
    cmip_84_05_ds = xr.open_dataset(
        "_Data/cmip5/pr_Amon_HadCM3_historical_r1i1p1_198412-200512.nc")
    cmip_ds = cmip_84_05_ds.merge(cmip_59_84_ds)  # in kg/m2/s
    cmip_ds = cmip_ds.assign_attrs(plot_legend="HadCM3 historical")
    cmip_ds = cmip_ds.rename({'pr': 'tp'})
    cmip_ds['tp'] *= 60 * 60 * 24  # to mm/day
    cmip_ds['time'] = standardised_time(cmip_ds)
    return cmip_ds


def standardised_time(dataset):
    """ Returns array of standardised times to plot """
    try:
        utime = dataset.time.values.astype(int)/(1e9 * 60 * 60 * 24 * 365)
    except Exception:
        time = np.array([d.strftime() for d in dataset.time.values])
        time2 = np.array([datetime.datetime.strptime(
            d, "%Y-%m-%d %H:%M:%S") for d in time])
        utime = np.array([d.timestamp() for d in time2]) / (60 * 60 * 24 * 365)
    return(utime + 1970)
