# Sampling

import os
import calendar
import datetime
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def random_location_sampler(df):
    """ Returns DataFrame of random location, apply to clean df only """

    df_squished = df.groupby(['latitude', 'longitude']).sum()
    df_s_reset = df_squished.reset_index()
    i = np.random.randint(len(df_s_reset), size=1)

    df_location = df_s_reset.iloc[i]
    lat = df_location['latitude'].values[0]
    lon = df_location['longitude'].values[0]

    df1 = df[df['latitude']==lat]
    df2 = df1[df1['longitude']==lon]

    return df2


def random_location_and_time_sampler(df, length=1000, seed=42): # TODO
    """ Returns DataFrame of random locations and times, apply to clean df only """
    
    np.random.seed(seed)
    i = np.random.randint(len(df), size=length)
    df_sampled = df.iloc[i]

    return df_sampled
