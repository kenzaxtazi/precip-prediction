# Data Preparation

import os
import calendar
import datetime
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import DataDownloader as dd
import Sampling as sa
import Clustering as cl


# Filepaths and URLs

mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'


def cumulative_monthly(da):
    """ Multiplies monthly averages by the number of day in each month """
    x, y, z = np.shape(da.values)
    times = np.datetime_as_string(da.time.values)
    days_in_month = []
    for t in times:
        year = t[0:4]
        month = t[5:7]
        days = calendar.monthrange(int(year), int(month))[1]
        days_in_month.append(days)
    dim = np.array(days_in_month)
    dim_mesh = np.repeat(dim, y*z).reshape(x, y, z) 
        
    return da * dim_mesh


def multivariate_data_prep(number=None, EDA_average=False, coords=None):
    """ 
    Outputs test, validation and training data for total precipitation as a function of time, 2m dewpoint temperature, 
    angle of sub-gridscale orography, orography, slope of sub-gridscale orography, total column water vapour,
    Nino 3.4, Nino 4 and NAO index for a single point.

    Inputs
        None

    Outputs
        x_train: training feature vector, numpy array 
        y_train: training output vector, numpy array
        x_test: testing feature vector, numpy array
        y_test: testing output vector, numpy array
    """
    if number != None:
        da_ensemble = dd.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop('number')
    if EDA_average == True:
        da_ensemble = dd.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim='number')
    else:
        da = dd.download_data(mask_filepath, xarray=True)

    if coords == None: 
        multiindex_df = da.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df_location = sa.random_location_sampler(df_clean)
        df = df_location.drop(columns=['latitude', 'longitude', 'slor', 'z'])  # 'anor',
    
    else: 
        da_location = da.interp(coords={'latitude':coords[0], 'longitude':coords[1]}, method='nearest')
        multiindex_df = da_location.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df = df_clean.drop(columns=['latitude', 'longitude', 'slor', 'z'])  # 'anor',

    df['time'] = df['time'].astype('int')
    df['time'] = (df['time'] - df['time'].min())/ (1e9*60*60*24*365)  # to years
    df['tp'] = df['tp']*1000  # to mm
    
    # Keep first of 70% for training
    train_df = df[ df['time']< df['time'].max()*0.7]
    xtrain = train_df.drop(columns=['tp']).values
    ytrain = train_df['tp'].values

    # Last 30% for evaluation
    eval_df = df[ df['time']> df['time'].max()*0.7]
    x_eval = eval_df.drop(columns=['tp']).values
    y_eval = eval_df['tp'].values

    # Training and validation data
    xval, xtest, yval, ytest = train_test_split(x_eval, y_eval, test_size=0.3333, shuffle=False)
    
    return xtrain, xval, xtest, ytrain, yval, ytest


def random_multivariate_data_prep(number=None,  EDA_average=False, length=3000, cluster_mask=None, seed=42):
    """ 
    Outputs test, validation and training data for total precipitation as a function of time, 2m dewpoint temperature, 
    angle of sub-gridscale orography, orography, slope of sub-gridscale orography, total column water vapour,
    Nino 3.4 index for given number randomly sampled data points.

    Inputs
        number, optional: specify desired ensemble run, integer
        length, optional: specify number of points to sample, integer
        seed, optional: specify seed, integer

    Outputs
        x_train: training feature vector, numpy array 
        y_train: training output vector, numpy array
        x_test: testing feature vector, numpy array
        y_test: testing output vector, numpy array
    """

    if number != None:
        da_ensemble = dd.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop('number')
        
    if EDA_average == True:
        da_ensemble = dd.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim='number')
    else:
        da = dd.download_data(mask_filepath, xarray=True)

    if cluster_mask != None:
        mask = xr.open_dataset(cluster_mask)
        mask_da = mask.overlap
        da = da.where(mask_da > 0, drop=True)
        
    multiindex_df = da.to_dataframe()
    df_clean = multiindex_df.dropna().reset_index()
    df_location = sa.random_location_and_time_sampler(df_clean, length=length, seed=seed)
    df = df_location

    df['time'] = df['time'].astype('int')
    df['time'] = (df['time'] - df['time'].min())/ (1e9*60*60*24*365)
    df['tp'] = df['tp']*1000  # to mm
    
    # Keep first of 70% for training
    train_df = df[ df['time']< df['time'].max()*0.7]
    xtrain = train_df.drop(columns=['tp']).values
    ytrain = train_df['tp'].values

    # Last 30% for evaluation
    eval_df = df[ df['time']> df['time'].max()*0.7]
    x_eval = eval_df.drop(columns=['tp']).values
    y_eval = eval_df['tp'].values

    # Training and validation data
    xval, xtest, yval, ytest = train_test_split(x_eval, y_eval, test_size=0.3333, shuffle=True)
    
    return xtrain, xval, xtest, ytrain, yval, ytest


def normalise(df):
    """ Normalise dataframe """

    features = list(df)
    for f in features:
        df[f] = df[f]/df[f].max()

    return df






    

