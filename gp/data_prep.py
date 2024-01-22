# Data Preparation

import sys
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split

# custom libraries
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/')  # noqa
from load import era5, location_sel
import gp.sampling as sa


def point_model(location:str, number:int=None, EDA_average:bool=False, maxyear:str=None)-> tuple:
    """
    Outputs test, validation and training data for total precipitation
    as a function of time, 2m dewpoint temperature, angle of sub-gridscale
    orography, orography, slope of sub-gridscale orography, total column
    water vapour, Nino 3.4, Nino 4 and NAO index for a single point.

    Args:
        location (str): specify if you want a specific location, list of 
            floats
        number (int, optional): specify desired ensemble run. Defaults to 
            None.
        EDA_average (bool, optional): specify if you want average of low 
            resolution ensemble runs. Defaults to False.
        maxyear (str, optional): _description_. Defaults to None.

    Returns:
        tuple:
            x_train (np.array): training feature vector
            y_train: training output vector
            x_test: testing feature vector
            y_test: testing output vector
            lmbda: lambda value for boxcox transformation
    """
    if maxyear is None:
        maxyear = '2020'

    if isinstance(location, str) == True:
        
        if number is not None:
            da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
            da = da_ensemble.sel(number=number).drop("number")
        if EDA_average is True:
            da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
            da = da_ensemble.mean(dim="number")
        else:
            da = era5.download_data(location, xarray=True)

        multiindex_df = da.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df_location = sa.random_location_sampler(df_clean)
        df = df_location.drop(columns=["lat", "lon", "slor", "anor", "z"])

    if isinstance(location, np.ndarray) == True:
        da_location = era5.collect_ERA5(
            (location[0], location[1]), minyear='1970', maxyear=maxyear)
        multiindex_df = da_location.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df = df_clean.drop(columns=["lat", "lon", "slor", "anor", "z"])

    df["time"] = pd.to_datetime(df["time"])
    df["time"] = pd.to_numeric(df["time"])
    # df["tp"] = log_transform(df["tp"])
    df = df[["time", "d2m", "tcwv", "N34", "tp"]]  # format order

    # Keep first of 70% for training
    x = df.drop(columns=["tp"]).values
    df[df['tp'] <= 0.0] = 0.0001
    print(df['tp'].min())
    y = df['tp'].values

    # Last 30% for evaluation
    xtrain, x_eval, ytrain, y_eval = train_test_split(
        x, y, test_size=0.3, shuffle=False)

    # Training and validation data
    xval, xtest, yval, ytest = train_test_split(
        x_eval, y_eval, test_size=1./3., shuffle=False)

    # Transformations
    ytrain_tr, lmbda = sp.stats.boxcox(ytrain)
    yval_tr = sp.stats.boxcox(yval, lmbda=lmbda)
    ytest_tr = sp.stats.boxcox(ytest, lmbda=lmbda)

    return xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr, lmbda


def areal_model(location, number=None, EDA_average=False, length=3000, seed=42,
                maxyear=None):
    """
    Outputs test, validation and training data for total precipitation as a
    function of time, 2m dewpoint temperature, angle of sub-gridscale
    orography, orography, slope of sub-gridscale orography, total column water
    vapour, Nino 3.4 index for given number randomly sampled data points
    for a given basin.

    Inputs
        location: specify area to train model
        number, optional: specify desired ensemble run, integer
        EDA_average, optional: specify if you want average of low resolution
            ensemble runs, boolean
        length, optional: specify number of points to sample, integer
        seed, optional: specify seed, integer

    Outputs
        x_train: training feature vector, numpy array
        y_train: training output vector, numpy array
        x_test: testing feature vector, numpy array
        y_test: testing output vector, numpy array
    """

    if number is not None:
        da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop("number")

    if EDA_average is True:
        da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = era5.download_data(location, xarray=True)

    # apply mask
    mask_filepath = location_sel.find_mask(location)
    masked_da = location_sel.apply_mask(da, mask_filepath)

    if maxyear is not None:
        df["time"] = df[df["time"] < maxyear]

    multiindex_df = masked_da.to_dataframe()
    df_clean = multiindex_df.dropna().reset_index()
    df = sa.random_location_and_time_sampler(
        df_clean, length=length, seed=seed)

    df["time"] = pd.to_datetime(df["time"])
    df["time"] = pd.to_numeric(df["time"])
    # df["tp"] = log_transform(df["tp"])
    df = df[["time", "lat", "lon", "d2m", "tcwv", "N34", "tp"]]  # format order

    # Keep first of 70% for training
    x = df.drop(columns=["tp"]).values
    df[df['tp'] <= 0.0] = 0.0001
    print(df['tp'].min())
    y = df['tp'].values

    # Last 30% for evaluation
    xtrain, x_eval, ytrain, y_eval = train_test_split(
        x, y, test_size=0.3, shuffle=False)

    # Training and validation data
    xval, xtest, yval, ytest = train_test_split(
        x_eval, y_eval, test_size=1./3., shuffle=True)

    # Transformations
    ytrain_tr, l = sp.stats.boxcox(ytrain)
    yval_tr = sp.stats.boxcox(yval, lmbda=l)
    ytest_tr = sp.stats.boxcox(ytest, lmbda=l)

    return xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr, l


def areal_model_eval(location, number=None, EDA_average=False, length=3000,
                     seed=42, minyear=1979, maxyear=2020):
    """
    Returns data to evaluate an areal model at a given location, area and time
    period.

    Variables:
        Total precipitation as a function of time, 2m dewpoint
        temperature, angle of sub-gridscale orography, orography, slope of
        sub-gridscale orography, total column water vapour, Nino 3.4, Nino 4
        and NAO index for a single point.

    Inputs:
        number, optional: specify desired ensemble run, integer
        EDA_average, optional: specify if you want average of low resolution
            ensemble runs, boolean
        coords [latitude, longitude], optional: specify if you want a specific
            location, list of floats
        mask, optional: specify area to train model, defaults to Upper Indus
            Basin

    Outputs
        x_tr: evaluation feature vector, numpy array
        y_tr: evaluation output vector, numpy array
    """
    if number is not None:
        da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop("number")
    if EDA_average is True:
        da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = era5.download_data(location, xarray=True)

    sliced_da = da.sel(time=slice(minyear, maxyear))

    if isinstance(location, str) is True:
        mask_filepath = location_sel.find_mask(location)
        masked_da = location_sel.apply_mask(sliced_da, mask_filepath)
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

    df["time"] = df["time"] - 1970  # to years
    #df["tp"] = log_transform(df["tp"])
    df = df[["time", "lat", "lon", "slor", "anor", "z",
             "d2m", "tcwv", "N34", "tp"]]  # format order

    xtr = df.drop(columns=["tp"]).values
    ytr = df["tp"].values

    return xtr, ytr


def average_over_coords(ds):
    """ Take average over latitude and longitude """
    ds = ds.mean("lon")
    ds = ds.mean("lat")
    return ds
