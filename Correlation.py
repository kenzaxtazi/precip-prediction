# Correlation

import datetime
import urllib

import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import DataExploration as de
import DataDownloader as dd
import FileDownloader as fd
import Clustering as cl


### Filepaths
mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'
eof_filepath = '/gws/nopw/j04/bas_climate/users/ktazi/z200_EOF2.nc'

def correlation_heatmap():

    ### Upper Indus Basin correlation plot 
    df = dd.download_data(mask_filepath)

    '''
    # create lags
    df['CGTI-1'] = df['CGTI'].shift(periods=1)
    df['CGTI-2'] = df['CGTI'].shift(periods=2)
    df['CGTI-3'] = df['CGTI'].shift(periods=3)
    df['CGTI-4'] = df['CGTI'].shift(periods=4)
    df['CGTI-5'] = df['CGTI'].shift(periods=5)
    df['CGTI-6'] = df['CGTI'].shift(periods=6)
    
    df['N34-1'] = df['N34'].shift(periods=1)
    df['N34-2'] = df['N34'].shift(periods=2)
    df['N34-3'] = df['N34'].shift(periods=3)
    df['N34-4'] = df['N34'].shift(periods=4)
    df['N34-5'] = df['N34'].shift(periods=5)
    df['N34-6'] = df['N34'].shift(periods=6)

    df['N4-1'] = df['N4'].shift(periods=1)
    df['N4-2'] = df['N4'].shift(periods=2)
    df['N4-3'] = df['N4'].shift(periods=3)
    df['N4-4'] = df['N4'].shift(periods=4)
    df['N4-5'] = df['N4'].shift(periods=5)
    df['N4-6'] = df['N4'].shift(periods=6)
    '''
    df = df.drop(columns=['expver', 'time'])
    df_clean = df.dropna()
    df_sorted = df_clean.sort_index(axis=1)

    # Correlation matrix
    corr = df_sorted.corr()  

    # Plot
    sns.set(style="white")

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    mask = np.triu(np.ones_like(corr, dtype=np.bool))  # generate a mask for the upper triangle
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Correlation plot for Upper Indus Basin')
    
    ### Cluster correlation plot 

    cluster_masks = ['Khyber_mask.nc', 'Gilgit_mask.nc', 'Ngari_mask.nc']
    names =['Gilgit regime', 'Ngari regime', 'Khyber regime']

    for i in range(3):
        cluster_df = dd.download_data(cluster_masks[i])

        # create lags
        cluster_df['CGTI-1'] = cluster_df['CGTI'].shift(periods=1)
        cluster_df['CGTI-2'] = cluster_df['CGTI'].shift(periods=2)
        cluster_df['CGTI-3'] = cluster_df['CGTI'].shift(periods=3)
        cluster_df['CGTI-4'] = cluster_df['CGTI'].shift(periods=4)
        cluster_df['CGTI-5'] = cluster_df['CGTI'].shift(periods=5)
        cluster_df['CGTI-6'] = cluster_df['CGTI'].shift(periods=6)
        ''''
        df_combined['N34-1'] = df_combined['N34'].shift(periods=1)
        df_combined['N34-2'] = df_combined['N34'].shift(periods=2)
        df_combined['N34-3'] = df_combined['N34'].shift(periods=3)
        df_combined['N34-4'] = df_combined['N34'].shift(periods=4)
        df_combined['N34-5'] = df_combined['N34'].shift(periods=5)
        df_combined['N34-6'] = df_combined['N34'].shift(periods=6)

        df_combined['N4-1'] = df_combined['N4'].shift(periods=1)
        df_combined['N4-2'] = df_combined['N4'].shift(periods=2)
        df_combined['N4-3'] = df_combined['N4'].shift(periods=3)
        df_combined['N4-4'] = df_combined['N4'].shift(periods=4)
        df_combined['N4-5'] = df_combined['N4'].shift(periods=5)
        df_combined['N4-6'] = df_combined['N4'].shift(periods=6)
        '''
        df = cluster_df(columns=['expver', 'time'])
        df_clean = df.dropna()
        df_sorted = df_clean.sort_index(axis=1)

        # Correlation matrix
        corr = df_sorted.corr()  

        # Plot
        sns.set(style="white")

        f, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        mask = np.triu(np.ones_like(corr, dtype=np.bool))  # generate a mask for the upper triangle
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

        plt.title(names[i] + '\n')

    plt.show()

def eof_correlation(eof_filepath, mask_filepath):
    ''' Returns plot and DataArray of areas with p<0.05 '''
    
    print('processing precipitation')
    da = dd.download_data(mask_filepath, xarray=True)
    tp_ds = da.mean(dim=['latitude', 'longitude']).tp
    tp = tp_ds.assign_coords(time=(tp_ds.time.astype('datetime64')))  
    tp_df = tp.to_dataframe()

    print('processing EOF')
    eof_da = xr.open_dataset(eof_filepath)
    eof_ds = eof_da.EOF
    eof = eof_ds.assign_coords(time=(eof_ds.time.astype('datetime64')))  
    eof_df = eof.to_dataframe()
    eof_pv = pd.pivot_table(eof_df, values='EOF', index= ['time'], columns=['latitude', 'longitude'])
    eof_reset = eof_pv.reset_index()
    eof_reset['time'] -= np.timedelta64(12,'h')

    print('combining')
    df_combined = pd.merge_ordered(tp_df, eof_reset, on='time')
    df_clean = df_combined.dropna()

    corr = df_clean.corr()

    corr.to_csv('Data/EOF_corr.csv')




