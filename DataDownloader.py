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
        all_var: boolean
    
    Returns 
        df: DataFrame of data, or
        ds: DataArray of data
    """

    path = 'Data/'
    now = datetime.datetime.now()

    if ensemble == True:
        filename = 'combi_data_ensemble' + '_' + now.strftime("%m-%Y")+'.csv'
    if all_var == True:
        filename = 'all_data' + '_' + now.strftime("%m-%Y")+'.csv'
    elif ensemble == False:
        filename = 'combi_data' + '_' + now.strftime("%m-%Y")+'.csv'

    filepath = path + filename
    print(filepath)

    if not os.path.exists(filepath):

        # Orography, humidity, precipitation and indices
        cds_df = cds_downloader(mask_filepath, ensemble=ensemble, all_var=all_var)
        ind_df = indice_downloader(all_var = all_var)
        df_combined = pd.merge_ordered(cds_df, ind_df, on='time', suffixes=("", "_y"))
        
        
        # Other variables not used in the GP
        if all_var == True:           
            mean_df = mean_downloader()
            uib_eofs_df = eof_downloader(mask_filepath, all_var = all_var)
    
            # Combine
            df_combined = pd.merge_ordered(df_combined, mean_df, on='time')
            df_combined = pd.merge_ordered(df_combined, uib_eofs_df, on=['time', 'latitude', 'longitude'])

        # Format and save
        df_clean = df_combined.dropna().drop('expver', axis=1)
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


def mean_downloader():

    def mean_formatter(filepath, coords=None, name=None):
        """ Returns dataframe averaged data over a optionally given area """

        da = xr.open_dataset(filepath)
        
        if 'expver' in list(da.dims):
            da = da.sel(expver=1)
            da = da.drop(['expver'])
        
        if coords != None:
            da = da.sel(latitude=slice(coords[0], coords[2]), longitude=slice(coords[1], coords[3]))

        mean_da = da.mean(dim=['longitude', 'latitude'], skipna=True) 
        clean_da = mean_da.assign_coords(time=(mean_da.time.astype('datetime64'))) 
        multiindex_df = clean_da.to_dataframe()
        df = multiindex_df #.reset_index()
        if name != None:
            df.rename(columns = {'EOF': name}, inplace=True)

        return df

    # Temperature
    temp_filepath = fd.update_cds_monthly_data(variables=['2m_temperature'], area=[40, 65, 20, 85], qualifier='temp')
    temp_df = mean_formatter(temp_filepath)

    # EOFs for 200hPa
    eof1_z200_c = mean_formatter('Data/regional_z200_EOF1.nc', coords = [40, 60, 35,70], name='EOF200C1')
    eof1_z200_b = mean_formatter('Data/regional_z200_EOF1.nc', coords = [19, 83, 16, 93], name='EOF200B1')
    eof2_z200_c = mean_formatter('Data/regional_z200_EOF2.nc', coords = [40, 60, 35,70], name='EOF200C2')
    eof2_z200_b = mean_formatter('Data/regional_z200_EOF2.nc', coords = [19, 83, 16, 93], name='EOF200B2')

    # EOFs for 500hPa
    eof1_z500_c = mean_formatter('Data/regional_z500_EOF1.nc', coords = [40, 60, 35,70], name='EOF500C1')
    eof1_z500_b = mean_formatter('Data/regional_z500_EOF1.nc', coords = [19, 83, 16, 93], name='EOF500B1')
    eof2_z500_c = mean_formatter('Data/regional_z500_EOF2.nc', coords = [40, 60, 35,70], name='EOF500C2')
    eof2_z500_b = mean_formatter('Data/regional_z500_EOF2.nc', coords = [19, 83, 16, 93], name='EOF500B2')

    # EOFs for 850hPa
    eof1_z850_c = mean_formatter('Data/regional_z850_EOF1.nc', coords = [40, 60, 35,70], name='EOF850C1')
    eof1_z850_b = mean_formatter('Data/regional_z850_EOF1.nc', coords = [19, 83, 16, 93], name='EOF850B1')
    eof2_z850_c = mean_formatter('Data/regional_z850_EOF2.nc', coords = [40, 60, 35,70], name='EOF850C2')
    eof2_z850_b = mean_formatter('Data/regional_z850_EOF2.nc', coords = [19, 83, 16, 93], name='EOF850B2')

    eof_df = pd.concat([eof1_z200_b, eof1_z200_c, eof2_z200_b, eof2_z200_c, eof1_z500_b, eof1_z500_c, 
                        eof2_z500_b, eof2_z500_c, eof1_z850_b, eof1_z850_c, eof2_z850_b, eof2_z850_c], axis=1)

    mean_df = pd.merge_ordered(temp_df, eof_df, on='time')

    return mean_df


def eof_downloader(mask_filepath, all_var=False):

    def eof_formatter(filepath, mask_filepath, name=None):
        """ Returns DataFrame of EOF over UIB  """
        eof_da = apply_mask(filepath, mask_filepath)
        eof_ds = eof_da.EOF
        eof2 = eof_ds.assign_coords(time=(eof_ds.time.astype('datetime64'))) 
        eof_multiindex_df = eof2.to_dataframe()
        eof_df = eof_multiindex_df.dropna()
        eof_df.rename(columns = {'EOF': name}, inplace = True)
        return eof_df

    # EOF UIB
    eof1_z200_u = eof_formatter('Data/regional_z200_EOF1.nc', mask_filepath, name='EOF200U1')
    eof1_z500_u = eof_formatter('Data/regional_z500_EOF1.nc', mask_filepath, name='EOF500U1')
    eof1_z850_u = eof_formatter('Data/regional_z850_EOF1.nc', mask_filepath, name='EOF850U1')

    eof2_z200_u = eof_formatter('Data/regional_z200_EOF2.nc', mask_filepath, name='EOF200U2')
    eof2_z500_u = eof_formatter('Data/regional_z500_EOF2.nc', mask_filepath, name='EOF500U2')
    eof2_z850_u = eof_formatter('Data/regional_z850_EOF2.nc', mask_filepath, name='EOF850U2')

    uib_eofs = pd.concat([eof1_z200_u, eof2_z200_u, eof1_z500_u, eof2_z500_u, eof1_z850_u, eof2_z850_u], axis=1)

    return uib_eofs


def indice_downloader(all_var=False):
    """ Return indice Dataframe """

    nao_url = 'https://www.psl.noaa.gov/data/correlation/nao.data'
    n34_url =  'https://psl.noaa.gov/data/correlation/nina34.data'
    n4_url =  'https://psl.noaa.gov/data/correlation/nina4.data'

    n34_df = fd.update_url_data(n34_url, 'N34')

    if all_var == False:    
        ind_df = n34_df.astype('float64')
    else:
        nao_df = fd.update_url_data(nao_url, 'NAO')
        n4_df = fd.update_url_data(n4_url, 'N4')
        ind_df = n34_df.join([nao_df, n4_df])
    
    return ind_df


def cds_downloader(mask_filepath, ensemble=False, all_var=False):
    """ Return CDS Dataframe """

    if ensemble == False:
        cds_filepath = fd.update_cds_monthly_data()
    else:
        cds_filepath = fd.update_cds_monthly_data(product_type='monthly_averaged_ensemble_members')

    masked_da = apply_mask(cds_filepath, mask_filepath)
    multiindex_df = masked_da.to_dataframe()
    cds_df = multiindex_df.reset_index()

    '''
    if all_var == False:
        cds_df = cds_df.drop(['anor'], axis=1)
    '''
    return cds_df
