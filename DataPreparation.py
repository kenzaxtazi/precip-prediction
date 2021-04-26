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

def point_model(location, number=None, EDA_average=False):
    """
    Outputs test, validation and training data for total precipitation as a function of time, 2m dewpoint temperature,
    angle of sub-gridscale orography, orography, slope of sub-gridscale orography, total column water vapour,
    Nino 3.4, Nino 4 and NAO index for a single point.

    Inputs
        number, optional: specify desired ensemble run, integer
        EDA_average, optional: specify if you want average of low resolution ensemble runs, boolean
        coords [latitude, longitude], optional: specify if you want a specific location, list of floats
        mask_filepath, optional:

    Outputs
        x_train: training feature vector, numpy array
        y_train: training output vector, numpy array
        x_test: testing feature vector, numpy array
        y_test: testing output vector, numpy array
    """
    if number != None:
        da_ensemble = dd.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop("number")
    if EDA_average == True:
        da_ensemble = dd.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = dd.download_data(location, xarray=True)

    if location is str:
        multiindex_df = da.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df_location = sa.random_location_sampler(df_clean)
        df = df_location.drop(columns=["lat", "lon", "slor", "anor", "z"])

    else:
        da_location = da.interp(
            coords={"lat": location[0], "lon": location[1]}, method="nearest"
        )
        multiindex_df = da_location.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df = df_clean.drop(columns=["lat", "lon", "slor", "anor", "z"])

    df["time"] = df["time"] - 1970 # to years
    df["tp"] = log_transform(df["tp"])
    df = df[["time", "d2m", "tcwv", "N34", "tp"]] #format order

    # Keep first of 70% for training
    train_df = df[df["time"] < df["time"].max() * 0.7]
    xtrain = train_df.drop(columns=["tp"]).values
    ytrain = train_df["tp"].values

    # Last 30% for evaluation
    eval_df = df[df["time"] > df["time"].max() * 0.7]
    x_eval = eval_df.drop(columns=["tp"]).values
    y_eval = eval_df["tp"].values

    # Training and validation data
    xval, xtest, yval, ytest = train_test_split(
        x_eval, y_eval, test_size=0.3333, shuffle=False
    )

    return xtrain, xval, xtest, ytrain, yval, ytest


def areal_model(location,
    number=None, EDA_average=False, length=3000, seed=42
):
    """
    Outputs test, validation and training data for total precipitation as a function of time, 2m dewpoint temperature,
    angle of sub-gridscale orography, orography, slope of sub-gridscale orography, total column water vapour,
    Nino 3.4 index for given number randomly sampled data points for a given basin.

    Inputs
        location: specify area to train model
        number, optional: specify desired ensemble run, integer 
        EDA_average, optional: specify if you want average of low resolution ensemble runs, boolean
        length, optional: specify number of points to sample, integer
        seed, optional: specify seed, integer

    Outputs
        x_train: training feature vector, numpy array
        y_train: training output vector, numpy array
        x_test: testing feature vector, numpy array
        y_test: testing output vector, numpy array
    """

    if number != None:
        da_ensemble = dd.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop("number")

    if EDA_average == True:
        da_ensemble = dd.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = dd.download_data(location, xarray=True)

    # apply mask
    mask_filepath = find_mask(location)
    masked_da = dd.apply_mask(da, mask_filepath)

    multiindex_df = masked_da.to_dataframe()
    df_clean = multiindex_df.dropna().reset_index()
    df = sa.random_location_and_time_sampler(
        df_clean, length=length, seed=seed)

    df["time"] = df["time"] - 1970 
    df["tp"] = log_transform(df["tp"])
    df = df[["time", "lat", "lon", "slor", "anor", "z", "d2m", "tcwv", "N34", "tp"]]

    # Remove last 10% of time for testing
    test_df = df[df["time"] > df["time"].max() * 0.9]
    xtest = test_df.drop(columns=["tp"]).values
    ytest = test_df["tp"].values

    # Training and validation data
    tr_df = df[df["time"] < df["time"].max() * 0.9]
    xtr = tr_df.drop(columns=["tp"]).values
    ytr = tr_df["tp"].values

    xtrain, xval, ytrain, yval = train_test_split(
        xtr, ytr, test_size=0.30, shuffle=False
    )  # Training and validation data

    """
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
    """
    return xtrain, xval, xtest, ytrain, yval, ytest


def areal_model_eval(location, number=None, EDA_average=False, length=3000, seed=42, minyear=1979, maxyear=2020):
    """
    Returns data to evaluate an areal model at a given location, area and time period.
    Variables: total precipitation as a function of time, 2m dewpoint temperature, angle of 
    sub-gridscale orography, orography, slope of sub-gridscale orography, total column 
    water vapour, Nino 3.4, Nino 4 and NAO index for a single point.

    Inputs
        number, optional: specify desired ensemble run, integer 
        EDA_average, optional: specify if you want average of low resolution ensemble runs, boolean
        coords [latitude, longitude], optional: specify if you want a specific location, list of floats
        mask, optional: specify area to train model, defaults to Upper Indus Basin

    Outputs
        x_tr: evaluation feature vector, numpy array
        y_tr: evaluation output vector, numpy array
    """
    if number != None:
        da_ensemble = dd.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop("number")
    if EDA_average == True:
        da_ensemble = dd.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = dd.download_data(location, xarray=True)

    sliced_da = da.sel(time=slice(minyear, maxyear))

    if isinstance(location, str) == True:
        mask_filepath = find_mask(location)
        masked_da = dd.apply_mask(sliced_da, mask_filepath)
        multiindex_df = masked_da.to_dataframe()
        multiindex_df = da.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df = sa.random_location_and_time_sampler(
        df_clean, length=length, seed=seed)

    else:
        da_location = sliced_da.interp(
            coords={"lat": location[0], "lon": location[1]}, method="nearest"
        )
        multiindex_df = da_location.to_dataframe()
        df = multiindex_df.dropna().reset_index()

    df["time"] = df["time"] - 1970 # to years
    df["tp"] = log_transform(df["tp"])
    df = df[["time", "lat", "lon", "slor", "anor", "z", "d2m", "tcwv", "N34", "tp"]] #format order
    
    xtr = df.drop(columns=["tp"]).values
    ytr = df["tp"].values

    return xtr, ytr



def normal_transform(df, features):
    """ 
    Normalise dataframe    
    Inputs
        df: dataframe
        features: list of features to transform, list of strings
    Output:
        df: updated dataframe.
    """
    for f in features:
        df[f] = df[f] / df[f].max()
    return df

def log_transform(x):
    """ Log transformation  for total precipitation """
    tp_max = 0.02340308390557766
    y = np.log(x*(np.e-1)/tp_max + 1)
    return y

def inverse_log_transform(x):
    """ Inverse log transformation for total precipitation """
    tp_max = 0.02340308390557766
    y = (np.exp(x)-1) * tp_max / (np.e-1)  # np.log(df[f]*(np.e-1)/df[f].max() + 1)
    return y

def find_mask(location):
    """ Returns a mask filepath for given location """

    mask_dic = {'ngari':'Data/Masks/Ngari_mask.nc', 'khyber':'Data/Masks/Khyber_mask.nc', 
                'gilgit':'Data/Masks/Gilgit_mask.nc', 'uib':'Data/Masks/ERA5_Upper_Indus_mask.nc',
                'sutlej': 'Data/Masks/Sutlej_basin_overlap.nc', 'beas':'Data/Masks/Beas_basin_overlap.nc'}
    mask_filepath = mask_dic[location]
    return mask_filepath


def average_over_coords(ds):
    """ Take average over latitude and longitude """
    ds = ds.mean("lon")
    ds = ds.mean("lat")
    return ds
