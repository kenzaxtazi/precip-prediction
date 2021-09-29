import xarray as xr
import numpy as np
import datetime


def collect_CORDEX():
    """ Downloads data from CORDEX East Asia model """
    cordex_90_ds = xr.open_dataset(
        "_Data/cordex/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-"
        "HadRM3P_v1_mon_199001-199012.nc")
    cordex_91_00_ds = xr.open_dataset(
        "_Data/cordex/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-"
        "HadRM3P_v1_mon_199101-200012.nc")
    cordex_01_ds = xr.open_dataset(
        "_Data/cordex/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-"
        "HadRM3P_v1_mon_200101-201012.nc")
    cordex_02_11_ds = xr.open_dataset(
        "_Data/cordex/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-"
        "HadRM3P_v1_mon_201101-201111.nc")
    cordex_90_00_ds = cordex_90_ds.merge(cordex_91_00_ds)
    cordex_01_11_ds = cordex_01_ds.merge(cordex_02_11_ds)
    cordex_ds = cordex_01_11_ds.merge(cordex_90_00_ds)  # in kg/m2/s
    cordex_ds = cordex_ds.assign_attrs(
        plot_legend="CORDEX EA - MOHC-HadRM3P historical")
    cordex_ds = cordex_ds.rename_vars({'pr': 'tp'})
    cordex_ds['tp'] *= 60 * 60 * 24   # to mm/day
    cordex_ds['time'] = standardised_time(cordex_ds)

    return cordex_ds


def standardised_time(dataset):
    """Returns array of standardised times to plot."""
    try:
        utime = dataset.time.values.astype(int)/(1e9 * 60 * 60 * 24 * 365)
    except Exception:
        time = np.array([d.strftime() for d in dataset.time.values])
        time2 = np.array([datetime.datetime.strptime(
            d, "%Y-%m-%d %H:%M:%S") for d in time])
        utime = np.array([d.timestamp() for d in time2]) / (60 * 60 * 24 * 365)
    return(utime + 1970)
