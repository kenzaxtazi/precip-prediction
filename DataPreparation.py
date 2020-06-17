# Model Data Preparation

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

import FileDownloader as fd


# Filepaths and URLs
  
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'

'''
tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
tp_ensemble_filepath ='/Users/kenzatazi/Downloads/adaptor.mars.internal-1587987521.7367163-18801-5-5284e7a8-222a-441b-822f-56a2c16614c2.nc'
mpl_filepath = '/Users/kenzatazi/Downloads/era5_msl_monthly_1979-2019.nc'
'''

def download_data(mask_filepath, xarray=False, ensemble=False): # TODO include variables in pathname 
    """ 
    Downloads data for prepearation or analysis

    Inputs
        mask_filepath: string
        xarray: boolean
        ensemble: boolean 
    
    Returns 
        df: DataFrame of data, or
        ds: DataArray of data
    """

    nao_url = 'https://www.psl.noaa.gov/data/correlation/nao.data'
    n34_url =  'https://psl.noaa.gov/data/correlation/nina34.data'
    n4_url =  'https://psl.noaa.gov/data/correlation/nina4.data'

    path = '/Users/kenzatazi/Downloads/'

    now = datetime.datetime.now()
    if ensemble == False:
        filename = 'combi_data' + '_' + now.strftime("%m-%Y")+'.csv' 
    else:
        filename = 'combi_data_ensemble' + '_' + now.strftime("%m-%Y")+'.csv'

    filepath = path + filename
    print(filepath)

    if not os.path.exists(filepath):

        # Indices
        nao_df = fd.update_url_data(nao_url, 'NAO')
        n34_df = fd.update_url_data(n34_url, 'N34')
        n4_df = fd.update_url_data(n4_url, 'N4')
        ind_df = nao_df.join([n34_df, n4_df]).astype('float64')

        # Orography, humidity and precipitation
        if ensemble == False:
            cds_filepath = fd.update_cds_data()
        else:
            cds_filepath = fd.update_cds_data(product_type='monthly_averaged_ensemble_members')

        masked_da = apply_mask(cds_filepath, mask_filepath)
        multiindex_df = masked_da.to_dataframe()
        cds_df = multiindex_df.reset_index()

        # Combine
        df_combined = pd.merge_ordered(cds_df, ind_df, on='time')
        df_clean = df_combined.drop(columns= ['expver']).dropna()
        df_clean['time'] = df_clean['time'].astype('int')
        df_clean.to_csv(filepath)

        if xarray == True:
            df_multi = df_clean.set_index(['time', 'longitude', 'latitude'])
            ds = df_multi.to_xarray()
            return ds
        else:    
            return df_combined
    
    else:
        df = pd.read_csv(filepath)

        if xarray == True:
            df_multi = df.set_index(['time', 'longitude', 'latitude'])
            ds = df_multi.to_xarray()
            return ds
        else:    
            return df


def apply_mask(data_filepath, mask_filepath):
    """
    Opens NetCDF files and applies Upper Indus Basin mask to ERA 5 data.
    Inputs:
        Data filepath, NetCDF
        Mask filepath, NetCDF
    Return:
        A Data Array
    """
    da = xr.open_dataset(data_filepath)
    if 'expver' in list(da.dims):
        print('expver found')
        da = da.sel(expver=1)

    mask = xr.open_dataset(mask_filepath)
    mask_da = mask.overlap

    # slice in case step has not been performed at download stage
    sliced_da = da.sel(latitude=slice(38, 30), longitude=slice(71.25, 82.75))    
    
    UIB = sliced_da.where(mask_da > 0, drop=True)

    return UIB


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


def point_data_prep(): # TODO Link to CDS API, currently broken
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

    da = download_data(mask_filepath, xarray=True, ensemble=True)
    print(da)
    
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


def multivariate_data_prep(): # TODO generalise to ensemble data
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
    da = download_data(mask_filepath, xarray=True)
    gilgit = da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    multiindex_df = gilgit.to_dataframe()
    df_clean = multiindex_df.reset_index()

    # Seperate y and x
    y = df_clean['tp'].values*1000

    x1 = df_clean.drop(columns=['tp'])
    print(x1['time'])
    x1['time'] = (x1['time'] - x1['time'].min())/ (1e9*60*60*24*365)
    x = x1.values.reshape(-1, 9)

    xtrain, xval, xtest, ytrain, yval, ytest = kfold_split(x, y)
    
    return xtrain, xval, xtest, ytrain, yval, ytest


def gp_area_prep(mask_filepath):
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

    df_combined = download_data(mask_filepath)
    
    df = df_combined.drop(columns=['expver'])
    df['time'] = df['time'].astype('int')
    df_normalised = normalise(df)
    df_clean = df_normalised.dropna()

    # Seperate y and x
    y = df_clean['tp'].values*1000  # to mm

    x1 = df_clean.drop(columns=['tp'])
    x1['time'] = (x1['time'] - x1['time'].min())/ (1e9*60*60*24*365)
    x = x1.values.reshape(-1, 11)
    
    xtrain, xval, xtest, ytrain, yval, ytest = kfold_split(x, y)
    
    return xtrain, xval, xtest, ytrain, yval, ytest


def normalise(df):
    """ Normalise dataframe """

    features = list(df)
    for f in features:
        df[f] = df[f]/df[f].max()

    return df


def kfold_split(x, y, dy=None, folds=5):
    """ 
    Split values into training, validation and test sets.
    The training and validation set are prepared for k folding.

    Inputs
        x: array of features
        y: array of target values
        dy: array of uncertainty on target values
        folds: number of kfolds

    Outputs
        xtrain: feature training set
        xvalidation: feature validation set
        xtest: feature test set

        ytrain: target training set
        yvalidation: target validation set
        ytest: target test set

        dy_train: uncertainty training set
        dy_validation: uncertainty validation set
        dy_test: uncertainty test set
    """
    # Remove test set without shuffling
    xtr, xtest, ytr, ytest = train_test_split(x, y, test_size=0.15)
    
    kf = KFold(n_splits=folds)
    for train_index, test_index in kf.split(xtr):
        xtrain, xval = xtr[train_index], xtr[test_index]
        ytrain, yval = ytr[train_index], ytr[test_index]

    return  xtrain, xval, xtest, ytrain, yval, ytest



    

