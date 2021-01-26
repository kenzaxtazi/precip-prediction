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

mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"


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
    dim_mesh = np.repeat(dim, y * z).reshape(x, y, z)

    return da * dim_mesh


def slm_multivariate_data_prep(number=None, EDA_average=False, coords=None):
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
        da = da_ensemble.sel(number=number).drop("number")
    if EDA_average == True:
        da_ensemble = dd.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = dd.download_data(mask_filepath, xarray=True)

    if coords == None:
        multiindex_df = da.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df_location = sa.random_location_sampler(df_clean)
        df = df_location.drop(columns=["lat", "lon", "slor", "anor", "z"])

    else:
        da_location = da.interp(
            coords={"lat": coords[0], "lon": coords[1]}, method="nearest"
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
        da = da_ensemble.sel(number=number).drop("number")
    if EDA_average == True:
        da_ensemble = dd.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = dd.download_data(mask_filepath, xarray=True)

    if coords == None:
        multiindex_df = da.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df = sa.random_location_sampler(df_clean)

    else:
        da_location = da.interp(
            coords={"lat": coords[0], "lon": coords[1]}, method="nearest"
        )
        multiindex_df = da_location.to_dataframe()
        df = multiindex_df.dropna().reset_index()

    df["time"] = df["time"] - 1970 # to years
    df["tp"] = log_transform(df["tp"])
    df = df[["time", "lat", "lon", "slor", "anor", "z", "d2m", "tcwv", "N34", "tp"]] #format order

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


def random_multivariate_data_prep(
    number=None, EDA_average=False, length=3000, cluster_mask=None, seed=42
):
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
        da = da_ensemble.sel(number=number).drop("number")

    if EDA_average == True:
        da_ensemble = dd.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = dd.download_data(mask_filepath, xarray=True)

    if cluster_mask != None:
        mask = xr.open_dataset(cluster_mask)
        mask_da = mask.overlap
        da = da.where(mask_da > 0, drop=True)

    multiindex_df = da.to_dataframe()
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