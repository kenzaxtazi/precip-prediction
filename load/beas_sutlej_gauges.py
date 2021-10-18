"""
Raw gauge measurements from the Beas and Sutlej valleys
"""

import numpy as np
import pandas as pd
from math import floor, ceil


def gauge_download(station, minyear, maxyear):
    """
    Download and format raw gauge data

    Args:
        station (str): station name (capitalised)
    Returns
       df (pd.DataFrame): precipitation gauge values
    """
    filepath = '_Data/RawGauge_BeasSutlej_.xlsx'
    daily_df = pd.read_excel(filepath, sheet_name=station)
    # daily_df.dropna(inplace=True)

    # To months
    daily_df.set_index('Date', inplace=True)
    daily_df.index = pd.to_datetime(daily_df.index)
    clean_df = daily_df.dropna()
    df_monthly = clean_df.resample('M').sum()/30.436875
    df = df_monthly.reset_index()
    df['Date'] = df['Date'].values.astype(float)/365/24/60/60/1e9
    df['Date'] = df['Date'] + 1970

    all_station_dict = pd.read_csv('_Data/gauge_info.csv')

    # to xarray DataSet
    lat, lon, _elv = all_station_dict.loc[station]
    df_ind = df.set_index('Date')
    ds = df_ind.to_xarray()
    ds = ds.assign_attrs(plot_legend="Gauge data")
    ds = ds.assign_coords({'lon': lon})
    ds = ds.assign_coords({'lat': lat})
    ds = ds.rename({'Date': 'time'})

    # Standardise time resolution
    raw_maxyear = float(ds.time.max())
    raw_minyear = float(ds.time.min())
    time_arr = np.arange(floor(raw_minyear*12-1)/12 + 1. /
                         24., ceil(raw_maxyear*12-1)/12, 1./12.)
    ds['time'] = time_arr

    tims_ds = ds.sel(time=slice(minyear, maxyear))
    return tims_ds


def all_gauge_data(minyear, maxyear, threshold=None):
    """
    Download data between specified dates for all active stations between two
    dates.
    Can specify the treshold for the the total number of active days
    during that period:
    e.g. for 10 year period -> 4018 - 365 = 3653
    """

    filepath = "_Data/qc_sushiwat_observations_MGM.xlsx"
    daily_df = pd.read_excel(filepath)

    maxy_str = str(maxyear) + '-01-01'
    miny_str = str(minyear) + '-01-01'
    mask = (daily_df['Date'] >= miny_str) & (daily_df['Date'] < maxy_str)
    df_masked = daily_df[mask]

    df_masked.set_index('Date', inplace=True)
    for col in list(df_masked):
        df_masked[col] = pd.to_numeric(df_masked[col], errors='coerce')

    if threshold is not None:
        df_masked = df_masked.dropna(axis=1, thresh=threshold)

    df_masked.index = pd.to_datetime(df_masked.index)
    df_monthly = df_masked.resample('M').sum()/30.436875
    df = df_monthly.reset_index()
    df['Date'] = df['Date'].values.astype(float)/365/24/60/60/1e9 + 1970

    df_ind = df.set_index('Date')
    ds = df_ind.to_xarray()
    ds = ds.assign_attrs(plot_legend="Gauge data")
    ds = ds.rename({'Date': 'time'})

    # Standardise time resolution
    time_arr = np.arange(round(minyear) + 1./24., maxyear, 1./12.)
    ds['time'] = time_arr

    return ds
