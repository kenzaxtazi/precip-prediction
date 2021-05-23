# Multi fidelity modeling

import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import GPy

import DataDownloader as dd
import DataPreparation as dp


station_dict = {'Banjar': [31.65,77.34], 'Sainj': [31.77,76.933], 'Ganguwal': [31.25,76.486]}

def era5_download(station):
    """ 
    Download and format ERA5 data 
    
    Args:
        station (str): station name (capitalised)
    
    Returns
       ds (xr.DataSet): ERA values for precipitation
    """
    
    # Load data
    era5_beas_da = dd.download_data('beas', xarray=True)
    beas_ds = era5_beas_da.tp 
    
    # Interpolate at location
    lat, lon = station_dict[station]
    ds = beas_ds.interp(coords={"lon": lon, "lat": lat}, method="nearest")

    return ds


def gauge_download(station):
    """ 
    Download and format raw gauge data 
    
    Args:
        station (str): station name (capitalised)
    
    Returns
       df (pd.DataFrame): Precipitation gauge values
    """
    filepath= 'Data/RawGauge_BeasSutlej_.xlsx'
    daily_df= pd.read_excel(filepath, sheet_name=station)
    daily_df.dropna(inplace=True)

    # To months
    daily_df.set_index('Date', inplace=True)
    daily_df.index = pd.to_datetime(daily_df.index)
    df_monthly = daily_df.resample('M').sum()/30.436875
    df = df_monthly.reset_index()
    df['Date'] = df['Date'].values.astype(float)/365/24/60/60/1e9 +1970

    return df

