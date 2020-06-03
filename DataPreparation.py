# Model Data Preparation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import PrecipitationDataExploration as pde
import FileDownloader as fd
import Clustering as cl


# Filepaths and URLs
    
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'

nao_url = 'https://www.psl.noaa.gov/data/correlation/nao.data'
n34_url =  'https://psl.noaa.gov/data/correlation/nina34.data'
n4_url =  'https://psl.noaa.gov/data/correlation/nina4.data'

tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
tp_ensemble_filepath ='/Users/kenzatazi/Downloads/adaptor.mars.internal-1587987521.7367163-18801-5-5284e7a8-222a-441b-822f-56a2c16614c2.nc'
mpl_filepath = '/Users/kenzatazi/Downloads/era5_msl_monthly_1979-2019.nc'

# Apply mask
da = pde.apply_mask(tp_ensemble_filepath, mask_filepath)



def point_data_prep(da): # TODO Link to CDS API
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
    # Front indices
    nao_df = fd.update_url_data(nao_url, 'NAO')
    n34_df = fd.update_url_data(n34_url, 'N34')
    n4_df = fd.update_url_data(n4_url, 'N4')
    ind_df = nao_df.join([n34_df, n4_df]).astype('float64')


    # Orography, humidity and precipitation
    cds_filepath = fd.update_cds_data(variables=['2m_dewpoint_temperature', 'angle_of_sub_gridscale_orography', 
                                                'orography', 'slope_of_sub_gridscale_orography', 
                                                'total_column_water_vapour', 'total_precipitation'])
    masked_da = pde.apply_mask(cds_filepath, mask_filepath)
    gilgit = masked_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    multiindex_df = gilgit.to_dataframe()
    cds_df = multiindex_df.reset_index()

    # Combine
    df_combined = pd.merge_ordered(cds_df, ind_df, on='time')  
    df = df_combined.drop(columns=['expver', 'longitude', 'latitude'])
    df['time'] = df['time'].astype('int')
    df_clean = df.dropna()

    # Seperate y and x
    y = df_clean['tp'].values*1000

    x1 = df_clean.drop(columns=['tp'])
    x1['time'] = (x1['time'] - x1['time'].min())/ (1e9*60*60*24*365)
    x = x1.values.reshape(-1, 9)
    
    x_train = x[0:400]
    y_train = y[0:400]

    x_test = x[400:-2]
    y_test = y[400:-2]

    return x_train, y_train, x_test, y_test


def area_data_prep(mask_da):
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
    # Front indices
    nao_df = fd.update_url_data(nao_url, 'NAO')
    n34_df = fd.update_url_data(n34_url, 'N34')
    n4_df = fd.update_url_data(n4_url, 'N4')
    ind_df = nao_df.join([n34_df, n4_df]).astype('float64')


    # Orography, humidity and precipitation
    cds_filepath = fd.update_cds_data(variables=['2m_dewpoint_temperature', 'angle_of_sub_gridscale_orography', 
                                                'orography', 'slope_of_sub_gridscale_orography', 
                                                'total_column_water_vapour', 'total_precipitation'])
    masked_da = pde.apply_mask(cds_filepath, mask_filepath)
    multiindex_df = masked_da.to_dataframe()
    cds_df = multiindex_df.reset_index()

    # Combine
    df_combined = pd.merge_ordered(cds_df, ind_df, on='time')  
    df = df_combined.drop(columns=['expver'])
    df['time'] = df['time'].astype('int')
    df_clean = df.dropna()

    # Seperate y and x
    y = df_clean['tp'].values*1000

    x1 = df_clean.drop(columns=['tp'])
    x1['time'] = (x1['time'] - x1['time'].min())/ (1e9*60*60*24*365)
    x = x1.values.reshape(-1, 9)
    
    x_train = x[0:400]
    y_train = y[0:400]

    x_test = x[400:-2]
    y_test = y[400:-2]

    return x_train, y_train, x_test, y_test