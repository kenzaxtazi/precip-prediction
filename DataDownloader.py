# Data Downloader

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
#mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'


def download_data(mask_filepath, xarray=False, ensemble=False, all_var=False): 
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

    path = 'Data/'

    now = datetime.datetime.now()
    if ensemble == False:
        filename = 'combi_data' + '_' + now.strftime("%m-%Y")+'.csv'
    if all_var == True:
        filename = 'all_data' + '_' + now.strftime("%m-%Y")+'.csv'
    else:
        filename = 'combi_data_ensemble' + '_' + now.strftime("%m-%Y")+'.csv'

    filepath = path + filename
    print(filepath)

    if not os.path.exists(filepath):

        # Indices

        n34_df = fd.update_url_data(n34_url, 'N34')

        if all_var == False:    
            ind_df = n34_df.astype('float64')
        else:
            nao_df = fd.update_url_data(nao_url, 'NAO')
            n4_df = fd.update_url_data(n4_url, 'N4')
            ind_df = n34_df.join([nao_df, n4_df])
        
        if all_var == True:
            # Temperature
            temp_filepath = fd.update_cds_monthly_data(variables=['2m_temperature'], area=[40, 65, 20, 85], qualifier='temp')
            temp_da = xr.open_dataset(temp_filepath)
            if 'expver' in list(temp_da.dims):
                temp_da = temp_da.sel(expver=1)
            temp_mean_da = temp_da.mean(dim=['longitude', 'latitude'], skipna=True) 
            multiindex_df = temp_mean_da.to_dataframe()
            temp_df = multiindex_df.reset_index()

            # CGTI
            z200_filepath = fd.update_cds_monthly_data(variables=['geopotential'], pressure_level='200', area=[40, 60, 35,70], qualifier='z200')
            z200_da = xr.open_dataset(z200_filepath)
            if 'expver' in list(z200_da.dims):
                z200_da = z200_da.sel(expver=1)
            cgti_da = z200_da.mean(dim=['longitude', 'latitude'], skipna=True) 
            multiindex_df = cgti_da.to_dataframe()
            cgti_df = multiindex_df.reset_index()
            cgti_df = cgti_df.rename(columns={"z":"CGTI"})

            eof_da = apply_mask('Data/UIB_z200_EOF2.nc', mask_filepath)
            eof_ds = eof_da.EOF
            eof2 = eof_ds.assign_coords(time=(eof_ds.time.astype('datetime64'))) 
            eof_multiindex_df = eof2.to_dataframe()
            eof_df = eof_multiindex_df.reset_index()
            eof_df['time'] -= np.timedelta64(12,'h')

        # Orography, humidity and precipitation
        if ensemble == False:
            cds_filepath = fd.update_cds_monthly_data()
        else:
            cds_filepath = fd.update_cds_monthly_data(product_type='monthly_averaged_ensemble_members')

        masked_da = apply_mask(cds_filepath, mask_filepath)
        multiindex_df = masked_da.to_dataframe()
        cds_df = multiindex_df.reset_index()

        # Combine
        df_combined = pd.merge_ordered(cds_df, ind_df, on='time')
        if all_var == True:
            df_combined2 = pd.merge_ordered(df_combined, temp_df, on='time')
            df_combined3 = pd.merge_ordered(df_combined2, cgti_df, on='time')
            df_combined = pd.merge_ordered(df_combined3, eof_df, on='time')
        
        df_clean = df_combined.dropna() #columns=['expver_x', 'expver_y']
        df_clean['time'] = df_clean['time'].astype('int')
        df_clean = df_clean.astype('float64')
        df_clean.to_csv(filepath)

        if xarray == True:
            if ensemble == True:
                df_multi = df_clean.set_index(['time', 'longitude', 'latitude', 'number'])
            else:
                df_multi = df_clean.set_index(['time', 'longitude', 'latitude'])
            ds = df_multi.to_xarray()
            return ds
        else:    
            return df_clean
    
    else:
        df = pd.read_csv(filepath)
        df_clean = df.drop(columns=['Unnamed: 0'])

        if xarray == True:
            if ensemble == True:
                df_multi = df_clean.set_index(['time', 'longitude', 'latitude', 'number'])
            else:
                df_multi = df_clean.set_index(['time', 'longitude', 'latitude'])
            ds = df_multi.to_xarray()
            return ds
        else:    
            return df_clean


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