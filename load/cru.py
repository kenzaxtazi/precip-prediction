"""
CRU dataset
"""

import datetime
import xarray as xr
import numpy as np

import load.location_sel as ls


def collect_CRU(location, minyear, maxyear):
    """ Downloads interpolated data from CRU model"""
    cru_ds = xr.open_dataset("_Data/CRU/interpolated_cru_1901-2019.nc")

    if type(location) == str:
        loc_ds = ls.select_basin(cru_ds, location)
    else:
        lat, lon = location
        loc_ds = cru_ds.interp(
            coords={"lon": lon, "lat": lat}, method="nearest")

    tim_ds = loc_ds.sel(time=slice(minyear, maxyear))
    ds = tim_ds.assign_attrs(plot_legend="CRU")  # in mm/month
    return ds


def download():
    """ Returns CRU data on a 0.25° by 0.25° grid """
    extent = ls.basin_extent('indus')
    ds = xr.open_dataset("_Data/CRU/cru_ts4.04.1901.2019.pre.dat.nc")
    ds_cropped = ds.sel(
        lon=slice(extent[1], extent[3]), lat=slice(extent[2], extent[0]))
    ds_cropped['pre'] /= 30.437  # TODO apply proper function to get mm/day
    ds_cropped['time'] = standardised_time(ds_cropped)

    # Standardise time resolution
    maxyear = float(ds_cropped.time.max())
    minyear = float(ds_cropped.time.min())
    time_arr = np.arange(round(minyear) + 1./24., round(maxyear), 1./12.)
    ds_cropped['time'] = time_arr

    ds = ds_cropped.rename_vars({'pre': 'tp'})
    x = np.arange(70, 85, 0.25)
    y = np.arange(25, 35, 0.25)
    interp_ds = ds.interp(lon=x, lat=y, method="nearest")
    interp_ds.to_netcdf("_Data/CRU/interpolated_cru_1901-2019.nc")


def standardised_time(dataset):
    """ Returns array of standardised times"""
    try:
        utime = dataset.time.values.astype(int)/(1e9 * 60 * 60 * 24 * 365)
    except Exception:
        time = np.array([d.strftime() for d in dataset.time.values])
        time2 = np.array([datetime.datetime.strptime(
            d, "%Y-%m-%d %H:%M:%S") for d in time])
        utime = np.array([d.timestamp() for d in time2]) / (60 * 60 * 24 * 365)
    return(utime + 1970)
