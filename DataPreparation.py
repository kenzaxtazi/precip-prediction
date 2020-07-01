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

import CrossValidation as cv
import DataDownloader as dd
import Sampling as sa


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


def point_data_prep():
    """ 
    Outputs test and training data for total precipitation as a function of time from an ensemble of models for a single location

    Inputs
        da: DataArray 

    Outputs
        x_train: training feature vector, numpy array
        y_train: training output vector, numpy array
        dy_train: training standard deviation vector, numpy array
        x_test: testing feature vector, numpy array
        y_test: testing output vector, numpy array
        dy_test: testing standard deviation vector, numpy array

    """

    da = dd.download_data(mask_filepath, xarray=True, ensemble=True)
    
    std_da = da.std(dim='number')
    mean_da = da.mean(dim='number')

    gilgit_mean = mean_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    gilgit_std = std_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')

    multi_index_df_mean = gilgit_mean.to_dataframe()
    df_mean= multi_index_df_mean.reset_index()
    df_mean_clean = df_mean.dropna()
    df_mean_clean['time'] = df_mean_clean['time'].astype('int')

    multi_index_df_std = gilgit_std.to_dataframe()
    df_std = multi_index_df_std.reset_index()
    df_std_clean = df_std.dropna()
    df_std_clean['time'] = df_std_clean['time'].astype('int')

    y = df_mean_clean['tp'].values*1000
    dy = df_std_clean['tp'].values*1000

    x_prime = df_mean_clean['time'].values.reshape(-1, 1)
    x = (x_prime - x_prime[0])/ (1e9*60*60*24*365)
    
    x_train = x[0:400]
    y_train = y[0:400]
    dy_train = dy[0:400]

    x_test = x[400:-2]
    y_test = y[400:-2]
    dy_test = dy[400:-2]

    return x_train, y_train, dy_train, x_test, y_test, dy_test


def multivariate_data_prep(number=None, coords=None):
    """ 
    Outputs test and training data for total precipitation as a function of time, 2m dewpoint temperature, 
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
    if number == None:
        da = dd.download_data(mask_filepath, xarray=True)
    else:
        da = dd.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da.sel(number=number).drop('number')

    if coords == None: 
        multiindex_df = da.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df_location = sa.random_location_sampler(df_clean)
        df = df_location.drop(columns=['latitude', 'longitude'])
    
    else: 
        da_location = da.interp(coords={'latitude':coords[0], 'longitude':coords[1]}, method='nearest')
        multiindex_df = da_location.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df = df_clean.drop(columns=['latitude', 'longitude'])

    df['time'] = df['time'].astype('int')
    df['time'] = (df['time'] - df['time'].min())/ (1e9*60*60*24*365)
    df['tp'] = df['tp']*1000  # to mm
    
    # Remove last 10% of time for testing
    test_df = df[ df['time']> df['time'].max()*0.9]
    xtest = test_df.drop(columns=['tp']).values
    ytest = test_df['tp'].values

    # Training and validation data
    tr_df = df[ df['time']< df['time'].max()*0.9]
    xtr = tr_df.drop(columns=['tp']).values
    ytr = tr_df['tp'].values

    xtrain, xval, ytrain, yval = cv.simple_split(xtr, ytr)
    
    return xtrain, xval, xtest, ytrain, yval, ytest


def multivariate_cv_data_prep(number=None, coords=None):
    """ 
    Outputs test and training data for total precipitation as a function of time, 2m dewpoint temperature, 
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
    if number == None:
        da = dd.download_data(mask_filepath, xarray=True)
    else:
        da = dd.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da.sel(number=number).drop('number')

    if coords == None: 
        multiindex_df = da.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df_location = sa.random_location_sampler(df_clean)
        df = df_location.drop(columns=['latitude', 'longitude'])
    
    else: 
        da_location = da.interp(coords={'latitude':coords[0], 'longitude':coords[1]}, method='nearest')
        multiindex_df = da_location.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df = df_clean.drop(columns=['latitude', 'longitude'])

    df['time'] = df['time'].astype('int')
    df['time'] = (df['time'] - df['time'].min())/ (1e9*60*60*24*365)
    df['tp'] = df['tp']*1000  # to mm
    
    # Remove last 10% of time for testing
    test_df = df[ df['time']> df['time'].max()*0.9]
    xtest = test_df.drop(columns=['tp']).values
    ytest = test_df['tp'].values

    # Training and validation data
    tr_df = df[ df['time']< df['time'].max()*0.9]
    xtr = tr_df.drop(columns=['tp']).values
    ytr = tr_df['tp'].values
    
    return xtr, xtest, ytr, ytest


def random_multivariate_data_prep(number=None, length=3000, seed=42):
    """ 
    Outputs test and training data for total precipitation as a function of time, 2m dewpoint temperature, 
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
    if number == None:
        da = dd.download_data(mask_filepath, xarray=True)
    else:
        da = dd.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da.sel(number=number).drop('number')

    multiindex_df = da.to_dataframe()
    df_clean = multiindex_df.dropna().reset_index()
    df_location = sa.random_location_and_time_sampler(df_clean, length=length, seed=seed)
    df = df_location

    df['time'] = df['time'].astype('int')
    df['time'] = (df['time'] - df['time'].min())/ (1e9*60*60*24*365)
    df['tp'] = df['tp']*1000  # to mm
    
    # Remove last 10% of time for testing
    test_df = df[ df['time']> df['time'].max()*0.9]
    xtest = test_df.drop(columns=['tp']).values
    ytest = test_df['tp'].values

    # Training and validation data
    tr_df = df[ df['time']< df['time'].max()*0.9]
    xtr = tr_df.drop(columns=['tp']).values
    ytr = tr_df['tp'].values

    xtrain, xval, ytrain, yval = cv.simple_split(xtr, ytr)
    
    return xtrain, xval, xtest, ytrain, yval, ytest


def gp_area_prep(mask_filepath, number=None):
    """ 
    Outputs test and training data for total precipitation as a function of time, 2m dewpoint temperature, 
    angle of sub-gridscale orography, orography, slope of sub-gridscale orography, total column water vapour,
    Nino 3.4, Nino 4 and NAO index for a given area.

    Inputs
        mask_da: mask for area of interest, DataArray

    Outputs
        x_train: training feature vector, numpy array
        y_train: training output vector, numpy array
        x_test: testing feature vector, numpy array
        y_test: testing output vector, numpy array
    """
    if number == None:
        df = dd.download_data(mask_filepath)
    else:
        df = dd.download_data(mask_filepath, ensemble=True)
        df = df[df['number']== number].drop('number')
    
    df['time'] = df['time'].astype('int')
    df['time'] = (df['time'] - df['time'].min())/ (1e9*60*60*24*365)
    df['tp'] = df['tp']*1000  # to mm
    df_clean = df.dropna()

    # Remove last 10% of time for testing
    test_df = df_clean[ df_clean['time']> df_clean['time'].max()*0.9]
    xtest = test_df.drop(columns=['tp']).values
    ytest = test_df['tp'].values

    # Training and validation data
    tr_df = df_clean[ df_clean['time']< df_clean['time'].max()*0.9]
    xtr = tr_df.drop(columns=['tp']).values
    ytr = tr_df['tp'].values
    
    xtrain, xval, ytrain, yval = cv.simple_split(xtr, ytr)
    
    return xtrain, xval, xtest, ytrain, yval, ytest


def normalise(df):
    """ Normalise dataframe """

    features = list(df)
    for f in features:
        df[f] = df[f]/df[f].max()

    return df






    

