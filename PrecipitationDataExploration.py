
# Precipitation Data Exploration

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
import cartopy.feature as cf
import matplotlib.ticker as tck
import matplotlib.cm as cm
import calendar
import datetime
import os
import urllib

# open data
tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
mpl_filepath = '/Users/kenzatazi/Downloads/era5_msl_monthly_1979-2019.nc'
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'


def apply_mask(tp_filepath, mask_filepath):
    """
    Opens NetCDF files and applies Upper Indus Basin mask to ERA 5 precipitation data.
    Inputs:
        Data filepath, NetCDF
        Mask filepath, NetCDF
    Return:
        A Data Array
    """
    tp = xr.open_dataset(tp_filepath)
    tp_da = tp.tp

    mask = xr.open_dataset(mask_filepath)
    mask_da = mask.overlap 

    sliced_tp_da = tp_da.sel(latitude=slice(38, 30), longitude=slice(71.25, 82.75))    
    UIB = sliced_tp_da.where(mask_da > 0, drop=True)

    return UIB


def save_csv_from_url(url, saving_path):
	""" Downloads data from a url and saves it to a specified path. """
	response = urllib.request.urlopen(url)
	with open(saving_path, 'wb') as f:
		f.write(response.read())


def update_data(url, file, name):
    """ Import the dataset """
    
    # Only download CSV if not present locally
    if not os.path.exists(file):
        save_csv_from_url(url, file)
    
    # create and format DataFrame 
    df = pd.read_csv(file)
    df_split = df[list(df)[0]].str.split(expand=True)
    df_long = pd.melt(df_split, id_vars=[0], value_vars=np.arange(1,13), var_name='month', value_name=name)
    
    # Create a datetime column
    df_long['date'] = df_long[0].astype(str) + '-' + df_long['month'].astype(str)
    df_long['date'] = pd.to_datetime(df_long['date'], errors='coerce') 
    
    # Clean
    df_clean = df_long.dropna()
    df_sorted = df_clean.sort_values(by=['date'])
    df_final = df_sorted.set_index('date')

    return pd.DataFrame(df_final[name])


def map_1979(UIB):
    """ Map of cumulative precipitation for 1979 """

    UIB_cum = cumulative_montly(UIB)
    UIB_1979 = UIB_cum.sel(time=slice('1979-01-16T12:00:00', '1980-01-01T12:00:00'))
    UIB_1979_sum = UIB_1979.sum(dim='time')

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([55, 95, 15, 45])
    g = UIB_1979_sum.plot(cmap='magma_r', vmin=0.01, cbar_kwargs={'label': '\n Total precipitation [m]', 'extend':'neither'})
    g.cmap.set_under('white')
    ax.add_feature(cf.BORDERS)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title('Upper Indus Basin Total Precipitation (1979) \n \n')
    plt.show()


def change_maps(UIB, UIB_1979_sum):
    """ Maps of average annual change from 1979 to 1989, 1999, 2009 and 2019 """

    UIB_1989 = UIB.sel(time=slice('1989-01-01T12:00:00', '1990-01-01T12:00:00'))
    UIB_1989_sum = UIB_1989.sum(dim='time')* 30.42
    UIB_1989_change = UIB_1989_sum/ UIB_1979_sum - 1 

    UIB_1999 = UIB.sel(time=slice('1999-01-01T12:00:00', '2000-01-01T12:00:00'))
    UIB_1999_sum = UIB_1999.sum(dim='time')* 30.42
    UIB_1999_change = UIB_1999_sum/ UIB_1979_sum - 1

    UIB_2009 = UIB.sel(time=slice('2009-01-01T12:00:00', '2010-01-01T12:00:00'))
    UIB_2009_sum = UIB_2009.sum(dim='time')* 30.42
    UIB_2009_change = UIB_2009_sum/ UIB_1979_sum - 1

    UIB_2019 = UIB.sel(time=slice('2019-01-01T12:00:00', '2020-01-01T12:00:00'))
    UIB_2019_sum = UIB_2019.sum(dim='time')*30
    UIB_2019_change = UIB_2019_sum/ UIB_1979_sum - 1

    UIB_changes = xr.concat([UIB_1989_change, UIB_1999_change, UIB_2009_change, UIB_2019_change],
                            pd.Index(['1989', '1999', '2009', '2019'], name='year'))

    g= UIB_changes.plot(x='longitude', y='latitude', col='year', col_wrap=2,
                        subplot_kws={'projection': ccrs.PlateCarree()},
                        cbar_kwargs={'label': 'Precipitation change', 'format': tck.PercentFormatter(xmax=1.0)})

    for ax in g.axes.flat:
        ax.coastlines()
        ax.gridlines()
        ax.set_extent([55, 95, 15, 45])
        ax.add_feature(cf.BORDERS)

    plt.show()


def timeseries(tp_da): 
    """ Precipitation timeseries for Gilgit, Skardu and Leh"""

    gilgit = tp_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    skardu = tp_da.interp(coords={'longitude':75.5550, 'latitude':35.3197 }, method='nearest')
    leh = tp_da.interp(coords={'longitude':77.5706, 'latitude':34.1536 }, method='nearest') 

    fig, axs = plt.subplots(3, sharex=True, sharey=True)

    gilgit.plot(ax=axs[0])
    axs[0].set_title('Gilgit (35.8884°N, 74.4584°E, 1500m)')
    axs[0].set_xlabel(' ')
    axs[0].grid(True)

    skardu.plot(ax=axs[1])
    axs[1].set_title('Skardu (35.3197°N, 75.5550°E, 2228m)')
    axs[1].set_xlabel(' ')
    axs[1].grid(True)

    leh.plot(ax=axs[2])
    axs[2].set_title('Leh (34.1536°N, 77.5706°E, 3500m)')
    axs[2].set_xlabel('Year')
    axs[2].grid(True)

    plt.show()


def cumulative_montly(da):
    """ Multiplies monthly averages by the number of day in each month """
    times = np.datetime_as_string(da.time.values)
    days_in_month = []
    for t in times:
        year = t[0:4]
        month = t[5:7]
        days = calendar.monthrange(int(year), int(month))[1]
        days_in_month.append(days)
    dim = np.array(days_in_month)
    dim_mesh = np.repeat(dim, 25*39).reshape(488,25,39) 
        
    return da * dim_mesh


def zeros_in_data(da):
    """ 
    Map and bar chart of the zeros points in a data set.
    Input:
        Data Array
    Returns:
        Two Matplotlib figures
    """

    df = da.to_dataframe('Precipitation')
    
    # Bar chart
    reduced_df = (df.reset_index()).drop(labels = ['latitude', 'longitude'], axis=1)
    clean_df = reduced_df.dropna()

    zeros = clean_df[clean_df['Precipitation']<= 0.00000]
    zeros['ones'] = np.ones(len(zeros))
    zeros['time'] = pd.to_datetime(zeros['time'].astype(int)).dt.strftime('%Y-%m')
    gr_zeros = zeros.groupby('time').sum()

    gr_zeros.plot.bar(y='ones', rot=0, legend=False)
    plt.ylabel('Number of zeros points')
    plt.xlabel('')
    

    # Map chart
    reduced_df2 = (df.reset_index()).drop(labels = ['time'], axis=1)
    clean_df2 = reduced_df2.dropna()
    zeros2 = clean_df2[clean_df2['Precipitation']<= 0]
    zeros2['ones'] = np.ones(len(zeros2))
    reset_df = zeros2.set_index(['latitude', 'longitude']).drop(labels='Precipitation', axis=1)

    sum_df = reset_df.groupby(reset_df.index).sum()
    sum_df.index = pd.MultiIndex.from_tuples(sum_df.index, names=['latitude', 'longitude'])
    sum_df = sum_df.reset_index()

    East = sum_df[sum_df['longitude']> 75]
    West = sum_df[sum_df['longitude']< 75]

    East_pv = East.pivot(index='latitude', columns='longitude')
    East_pv = East_pv.droplevel(0, axis=1)
    East_da = xr.DataArray(data=East_pv)

    West_pv = West.pivot(index='latitude', columns='longitude')
    West_pv = West_pv.droplevel(0, axis=1)
    West_da = xr.DataArray(data=West_pv)

    """
    # make sure all the zeros are within the mask
    mask = xr.open_dataset(mask_filepath)
    mask_da = mask.overlap
    UIB_zeros = reduced_da2.where(mask_da > 0)
    """

    UIB_sum = (da.sum(dim='time'))
    UIB_outline = UIB_sum.where(UIB_sum <=0, -1)
    UIB_borders = UIB_sum.where(UIB_sum >0, -10)

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([70, 83, 30, 38])
    g = UIB_outline.plot(cmap='Blues_r', vmax=-0.5, vmin=-5, add_colorbar=False)
    g.cmap.set_over('white')
    w= West_da.plot(cmap='magma_r', vmin=1.00, cbar_kwargs={'label': '\n Number of zero points', 'extend':'neither'})
    w.cmap.set_under('white')
    e= East_da.plot(cmap='magma_r', vmin=1.00, add_colorbar=False)
    e.cmap.set_under('white')
    ax.add_feature(cf.BORDERS)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title('Zero points \n \n')
    plt.show()

