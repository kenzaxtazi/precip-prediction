# Data Preparation

import sys
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# custom libraries
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/')  # noqa
from load import era5, location_sel
import gp.sampling as sa


def point_model(location: str | np.ndarray, number:int=None, EDA_average:bool=False, maxyear:str=None, seed=42, all_var=False)-> tuple:
    """
    Output training, validation and test sets for total precipitation.

    Args:
        location (str | np.ndarray): 'uib', 'khyber', 'ngari', 'gilgit' or [lat, lon] coordinates.
        number (int, optional): ensemble run number. Defaults to None.
        EDA_average (bool, optional): use ERA5 low-res ensemble average. Defaults to False.
        maxyear (str, optional): end year (inclusive). Defaults to None.
        seed (int, optional): sampling random generator seed. Defaults to 42.
        all_var (bool, optional): all the variables studied if True or only final selection for paper if False. Defaults to False.

    Returns:
        tuple: contains
            x_train (np.array): training feature vector
            x_val (np.array): validation feature vector
            x_test (np.array): testing feature vector
            y_train (np.array): training output vector
            y_val (np.array): validation output vector
            y_test (np.array): testing output vector
            lmbda: lambda value for Box Cox transformation
    """

    # End year for data
    if maxyear is None:
        maxyear = '2020'

    # Variable list
    if all_var is True:
        var_list = ["time", "tcwv", "d2m", "EOF200U",  "t2m", "EOF850U",  "EOF500U", "EOF500B2", "EOF200B",
             "NAO", "EOF500U2", "N34", "EOF850U2", "EOF500B1", "tp",]
    else:
        var_list =["time", "tcwv", "d2m", "EOF200U",  "t2m", "EOF500U", "tp",]
    
    # Download data and format data for string location
    if isinstance(location, str) == True:
        if number is not None:
            da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
            da = da_ensemble.sel(number=number).drop("number")
        if EDA_average is True:
            da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
            da = da_ensemble.mean(dim="number")
        else:
            da = era5.download_data(location, xarray=True, all_var=True,)

        multiindex_df = da.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df_location = sa.random_location_sampler(df_clean)
        df = df_location.drop(columns=["lat", "lon", "slor", "anor", "z"])

    # Download data and format data for coordinates
    if isinstance(location, np.ndarray) == True:
        da_location = era5.collect_ERA5(
            (location[0], location[1]), minyear='1970', maxyear=maxyear, all_var=True)
        multiindex_df = da_location.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df = df_clean.drop(columns=["lat", "lon", "slor", "anor", "z"])

    # Standardize time and select variables
    df["time"] = pd.to_datetime(df["time"])
    df["time"] = pd.to_numeric(df["time"])
    df = df[var_list]  

    # Keep first of 70% for training
    x = df.drop(columns=["tp"]).values
    df[df['tp'] <= 0.0] = 0.0001
    y = df['tp'].values
    
    xtrain, x_eval, ytrain, y_eval = train_test_split(
        x, y, test_size=0.3, shuffle=False)

    # Last 30% for evaluation
    xval, xtest, yval, ytest = train_test_split(x_eval, 
        y_eval, test_size=1./3., shuffle=True, random_state=seed)

    # Precipitation transformation
    ytrain_tr, lmbda = sp.stats.boxcox(ytrain)
    yval_tr = sp.stats.boxcox(yval, lmbda=lmbda)
    ytest_tr = sp.stats.boxcox(ytest, lmbda=lmbda)

    # Features scaling
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xval = scaler.transform(xval)
    xtest = scaler.transform(xtest)

    return xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr, lmbda


def areal_model(location:str, number:int=None, EDA_average=False, length=3000, seed=42,
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
        da = era5.download_data(location, xarray=True, all_var=True)

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
    df = df[["tcwv", "slor", "d2m", "lon", "z", "EOF200U", "t2m", "EOF850U", "tp"]]  # format order

    # Keep first of 70% for training
    x = df.drop(columns=["tp"]).values
    df[df['tp'] <= 0.0] = 0.0001

    y = df['tp'].values

    # Last 30% for evaluation
    xtrain, x_eval, ytrain, y_eval = train_test_split(
        x, y, test_size=0.3, shuffle=False,)

    # Training and validation data
    xval, xtest, yval, ytest = train_test_split(x_eval, 
        y_eval, test_size=1./3., shuffle=True, random_state=seed)

    # Transformations
    ytrain_tr, l = sp.stats.boxcox(ytrain)
    yval_tr = sp.stats.boxcox(yval, lmbda=l)
    ytest_tr = sp.stats.boxcox(ytest, lmbda=l)

    return xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr, l


def areal_model_new(location, number=None, EDA_average=False, length=3000, seed=42,
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
        length, optional: specify number of points to sample for training, integer
        seed, optional: specify seed, integer

    Outputs
        x_train: training feature vector, numpy array
        y_train: training output vector, numpy array
        x_test: testing feature vector, numpy array
        y_test: testing output vector, numpy array
    """
    if maxyear is None:
        maxyear = '2020'

    if number is not None:
        da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop("number")

    if EDA_average is True:
        da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        ds = era5.collect_ERA5(location, minyear="1970", maxyear=maxyear, all_var=True)

    # apply mask
    mask_filepath = location_sel.find_mask(location)
    masked_ds = location_sel.apply_mask(ds, mask_filepath)

    train_ds = masked_ds.sel(time=slice('1970', '2005'))
    eval_ds = masked_ds.sel(time=slice('2005', '2020'))

    var_list = ["time", "lon", "lat", "tcwv", "slor", "d2m", "z", "EOF200U",  "t2m", "EOF850U",  "EOF500U", "EOF500B2", "EOF200B",
             "anor", "NAO", "EOF500U2", "N34", "EOF850U2", "EOF200B", "tp",]


    multiindex_df = train_ds.to_dataframe() 
    df_train = multiindex_df.dropna().reset_index()
    df_train["time"] = pd.to_datetime(df_train["time"])
    df_train["time"] = pd.to_numeric(df_train["time"])
    # df_train["tp"] = log_transform(df_train["tp"])
    #df_train = df_train[["time", "lat", "lon", "d2m", "tcwv", "N34", "tp"]]  # format order
    df_train = df_train[var_list]
    df_train[df_train['tp'] <= 0.0] = 0.0001

    # Sample training
    df_train_samp = sa.random_location_and_time_sampler(df_train, length=length, seed=seed)  
    xtrain = df_train_samp.drop(columns=["tp"]).values
    ytrain = df_train_samp['tp'].values

    # Evaluation data
    multiindex_df = eval_ds.to_dataframe() 
    df_eval = multiindex_df.dropna().reset_index()
    df_eval["time"] = pd.to_datetime(df_eval["time"])
    df_eval["time"] = pd.to_numeric(df_eval["time"])

    df_eval = df_eval[var_list]
    #df_eval[df_eval['tp'] <= 0.0] = 0.0001
    
    loc_df = df_eval.groupby(['lat','lon']).mean().reset_index()
    locs = loc_df[['lon','lat']].values

    xval_list = []
    yval_list = []
    xtest_list = []
    ytest_list = []  

    for i in range(len(locs)):
        lon, lat = locs[i]
        loc_df = df_eval[(df_eval['lat'] == lat) & (df_eval['lon'] == lon)]
        loc_df.dropna(inplace=True)
        loc_df.loc[loc_df['tp'] <= 0.0] = 0.0001
        xeval = loc_df.drop(columns=["tp"]).values
        yeval = loc_df['tp'].copy(deep=False).values
        loc_xval, loc_xtest, loc_yval, loc_ytest = train_test_split(xeval, yeval, test_size=1./3., shuffle=True, random_state=seed)
        xval_list.extend(loc_xval)
        yval_list.extend(loc_yval)
        xtest_list.extend(loc_xtest)
        ytest_list.extend(loc_ytest)
    
    xval = np.array(xval_list)
    yval = np.array(yval_list)
    xtest = np.array(xtest_list)
    ytest = np.array(ytest_list)

    # Training and validation data
    
    # Transformations
    ytrain_tr, l = sp.stats.boxcox(ytrain)
    yval_tr = sp.stats.boxcox(yval, lmbda=l)
    ytest_tr = sp.stats.boxcox(ytest, lmbda=l)

    return xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr, l


def areal_model_eval(location, lmbda, number=None, EDA_average=False, length=3000,
                     seed=42, minyear="1970", maxyear="2020"):
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
        da_ensemble = era5.download_data('uib', xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop("number")
    if EDA_average is True:
        da_ensemble = era5.download_data('uib', xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = era5.download_data('uib', xarray=True)

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

    df["time"] = pd.to_numeric(pd.to_datetime(df["time"])) # to years
    df["tp_tr"] = sp.stats.boxcox(df["tp"].values, lmbda=lmbda)
    df = df[["time", "lat", "lon", "d2m", "tcwv", "N34", "tp_tr"]]  # format order
    #df = df[["time", "lat", "lon", "slor", "anor", "z", "d2m", "tcwv", "N34", "tp_tr"]]  # format order
    
    xtr = df.drop(columns=["tp_tr"]).values
    ytr = df["tp_tr"].values

    return xtr, ytr


def average_over_coords(ds):
    """ Take average over latitude and longitude """
    ds = ds.mean("lon")
    ds = ds.mean("lat")
    return ds


def shared_evaluation_sets(location:str, length=3000, seed=42):

    uib_ds = era5.collect_ERA5('uib', minyear="2005", maxyear="2020")
    df_clean = uib_ds.to_dataframe().dropna()
    df = sa.random_location_and_time_sampler(df_clean, length=length, seed=seed)
    df_eval = df[["tcwv", "slor", "d2m", "lon", "z", "EOF200U", "t2m", "EOF850U", "tp"]]

    y_df = df_eval["tp"]
    x_df = df_eval.drop(columns=["tp"])
    xval_uib_df, xtest_uib_df, yval_uib_df, ytest_uib_df = train_test_split(x_df, y_df, test_size=1./3., shuffle=True, random_state=seed)

    if location == 'uib':
        xval_df = xval_uib_df.reset_index()
        xval_df['time'] = pd.to_datetime(xval_df['time'])
        xval_df['time'] = pd.to_numeric(xval_df['time'])
        xval = xval_df.values

        xtest_df = xtest_uib_df.reset_index()
        xtest_df['time'] = pd.to_datetime(xtest_df['time'])
        xtest_df['time'] = pd.to_numeric(xtest_df['time'])
        xtest = xtest_df.values

        yval, ytest =  yval_uib_df.values, ytest_uib_df.values

    if location in ('khyber', 'ngari', 'gilgit'):
        mask_filepath = location_sel.find_mask(location)

        xval_uib_df = xval_uib_df[~xval_uib_df.index.duplicated()]
        xtest_uib_df = xtest_uib_df[~xtest_uib_df.index.duplicated()]
        yval_uib_df = yval_uib_df[~yval_uib_df.index.duplicated()]
        ytest_uib_df = ytest_uib_df[~ytest_uib_df.index.duplicated()]
        
        xval_uib_ds = xval_uib_df.to_xarray()
        xtest_uib_ds = xtest_uib_df.to_xarray()
        yval_uib_ds = yval_uib_df.to_xarray()
        ytest_uib_ds = ytest_uib_df.to_xarray()

        xval_ds = location_sel.apply_mask(xval_uib_ds, mask_filepath)
        xtest_ds = location_sel.apply_mask(xtest_uib_ds, mask_filepath)
        yval_ds = location_sel.apply_mask(yval_uib_ds, mask_filepath)
        ytest_ds = location_sel.apply_mask(ytest_uib_ds, mask_filepath)

        xval_df = xval_ds.to_dataframe().reset_index().dropna()
        xtest_df = xtest_ds.to_dataframe().reset_index().dropna()
        yval_df = yval_ds.to_dataframe().dropna()
        ytest_df = ytest_ds.to_dataframe().dropna()

        xval_df['time'] = pd.to_datetime(xval_df['time'])
        xval_df['time'] = pd.to_numeric(xval_df['time'])
        xval = xval_df.values
        
        xtest_df['time'] = pd.to_datetime(xtest_df['time'])
        xtest_df['time'] = pd.to_numeric(xtest_df['time'])
        xtest = xtest_df.values

        yval, ytest =  yval_uib_df.values, ytest_uib_df.values

    if location is isinstance(tuple):
        xtest_uib_df = xtest_uib_df.reset_index()
        ytest_uib_df = ytest_uib_df.reset_index()
        xval_uib_df = xval_uib_df.reset_index()
        yval_uib_df = yval_uib_df.reset_index()

        xval_df = xval_uib_df[(xval_uib_df['lat']== location[0]) & (xval_uib_df['lon'] == location[1])]
        yval_df = yval_uib_df[(yval_uib_df['lat']== location[0]) & (yval_uib_df['lon'] == location[1])]
        xtest_df = xtest_uib_df[(xtest_uib_df['lat']== location[0]) & (xtest_uib_df['lon'] == location[1])]
        ytest_df = ytest_uib_df[(ytest_uib_df['lat']== location[0]) & (ytest_uib_df['lon'] == location[1])]

        xval = xval_df[["time", "tcwv", "d2m", "tp"]]
        yval = yval_df["tp"]
        xtest = xtest_df[["time", "tcwv", "d2m", "tp"]]
        ytest = ytest_df["tp"]

    return  xval, xtest, yval, ytest

