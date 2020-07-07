# Sampling

import os
import calendar
import datetime
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DataDownloader as dd

mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'


def random_location_sampler(df):
    """ Returns DataFrame of random location, apply to clean df only """

    df_squished = df[['latitude', 'longitude']].reset_index()
    df_s_reset = df_squished.drop_duplicates()
    i = np.random.randint(len(df_s_reset), size=1)

    df_location = df_s_reset.iloc[i]
    lat = df_location['latitude'].values[0]
    lon = df_location['longitude'].values[0]
    print('lat='+ str(lat)+', lon='+ str(lon))

    df1 = df[df['latitude']==lat]
    df2 = df1[df1['longitude']==lon]

    return df2


def random_location_generator(N=50, UIB=False):
    """ Returns DataFrame of random location, apply to clean df only """

    coord_list = []

    df = dd.download_data(mask_filepath)
    df_squished = df[['latitude', 'longitude']].reset_index()
    df_s_reset = df_squished.drop_duplicates()

    if UIB == True:
        coord_list = df_s_reset[['latitude', 'longitude']].values

    else: 
        indices = np.random.randint(len(df_s_reset), size=N)

        for i in indices: 
            df_location = df_s_reset.iloc[i]
            lat = df_location['latitude']
            lon = df_location['longitude']
            coord_list.append([lat,lon])

    return coord_list


def random_location_and_time_sampler(df, length=1000, seed=42):
    """ Returns DataFrame of random locations and times, apply to clean df only """
    
    np.random.seed(seed)
    i = np.random.randint(len(df), size=length)
    df_sampled = df.iloc[i]

    return df_sampled
