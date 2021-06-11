"""
CRU dataset
"""

import os
import glob
import calendar
import datetime
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DataDownloader as dd
import LocationSel as ls

def standardised_time(dataset):
    """ Returns array of standardised times to plot """
    try:
        utime = dataset.time.values.astype(int)/(1e9 * 60 * 60 * 24 * 365)
    except Exception:
        time = np.array([d.strftime() for d in dataset.time.values])
        time2 = np.array([datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in time])
        utime = np.array([d.timestamp() for d in time2])/ ( 60 * 60 * 24 * 365)
    return(utime + 1970)

def collect_CRU(location):
    """ Downloads data from CRU model"""
    cru_ds = xr.open_dataset("Data/cru_ts4.04.1901.2019.pre.dat.nc")
    cru_ds = cru_ds.assign_attrs(plot_legend="CRU") # in mm/month
    cru_ds = cru_ds.rename_vars({'pre': 'tp'})
    cru_basin = ls.select_basin(cru_ds, location, interpolate=True)
    cru_basin['tp'] /= 30.437  #TODO apply proper function to get mm/day
    cru_basin['time'] = standardised_time(cru_basin)
    return cru_basin
