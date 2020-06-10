
# Precipitation Data Exploration

import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.cm as cm
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import FileDownloader as fd
import DataPreparation as dp




data_filepath = fd.update_cds_data()
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'

'''
tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
mpl_filepath = '/Users/kenzatazi/Downloads/era5_msl_monthly_1979-2019.nc'
'''


def annual_map(data_filepath, mask_filepath, variable, year, cumulative=False):
    """ Map of cumulative precipitation for 1979 """

    da = dp.apply_mask(data_filepath, mask_filepath)
    da_var = da[variable]
    da_year = da_var.sel(time=slice(str(year)+'-01-16T12:00:00', str(year+1)+'-01-01T12:00:00'))

    if cumulative is True:
        da_processed= dp.cumulative_montly(da_year)
        da_final = da_processed.sum(dim='time')
    else:
        da_final = da_year.mean(dim='time')
   

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([55, 95, 15, 45])
    g = da_final.plot(cmap='magma_r', vmin=0.01, cbar_kwargs={'label': '\n Total precipitation [m]', 'extend':'neither'})
    g.cmap.set_under('white')
    ax.add_feature(cf.BORDERS)
    ax.text(64, 27, 'PAKISTAN')
    ax.text(63, 33, 'AFGHANISTAN')
    ax.text(80, 36, 'CHINA')
    ax.text(74, 28, 'INDIA')
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


def sample_timeseries(data_filepath, variable='tp', longname='Total precipitation [m/day]'):
    """ Timeseries for Gilgit, Skardu and Leh"""

    da = xr.open_dataset(data_filepath)
    if 'expver' in list(da.dims):
        print('expver found')
        da = da.sel(expver=1)

    ds = da[variable]
    
    gilgit = ds.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    skardu = ds.interp(coords={'longitude':75.5550, 'latitude':35.3197 }, method='nearest')
    leh = ds.interp(coords={'longitude':77.5706, 'latitude':34.1536 }, method='nearest') 

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


def nans_in_data(): # TODO
    """ Returns number of nans in data with plots for UIB """


def averaged_timeseries(data_filepath, mask_filepath, variable='tp', longname='Total precipitation [m/day]'):
    """ Timeseries for the Upper Indus Basin"""

    da = dp.apply_mask(data_filepath, mask_filepath)

    ds = da[variable]
    ds_timeseries = ds.groupby('time').mean()

    ds_timeseries.plot()
    plt.title('Upper Indus Basin')
    plt.ylabel(longname)
    plt.xlabel('Year')
    plt.grid(True)

    plt.show()


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


def spatial_autocorr(): # TODO
    """ Plots spatial autocorrelation """


def temp_autocorr(data_filepath, mask_filepath, variable='tp', longname='Total precipitation [m/day]'): # TODO
    """ Plots temporal autocorrelation """
    
    da = xr.open_dataset(data_filepath)
    if 'expver' in list(da.dims):
        print('expver found')
        da = da.sel(expver=1)

    ds = da[variable]
    
    gilgit = ds.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    skardu = ds.interp(coords={'longitude':75.5550, 'latitude':35.3197 }, method='nearest')
    leh = ds.interp(coords={'longitude':77.5706, 'latitude':34.1536 }, method='nearest')

    gilgit_df = gilgit.to_dataframe().dropna()
    skardu_df = skardu.to_dataframe().dropna()
    leh_df = leh.to_dataframe().dropna()

    fig, axs = plt.subplots(3, sharex=True, sharey=True)

    axs[0].acorr(gilgit_df[variable], usevlines=True, normed=True, maxlags=50, lw=2)
    axs[0].set_title('Gilgit (35.8884°N, 74.4584°E, 1500m)')
    axs[0].set_xlabel(' ')
    axs[0].grid(True)

    axs[1].acorr(skardu_df[variable], usevlines=True, normed=True, maxlags=50, lw=2)
    axs[1].set_title('Skardu (35.3197°N, 75.5550°E, 2228m)')
    axs[1].set_xlabel(' ')
    axs[1].grid(True)

    a = axs[2].acorr(leh_df[variable], usevlines=True, normed=True, maxlags=50, lw=2)
    axs[2].set_title('Leh (34.1536°N, 77.5706°E, 3500m)')
    axs[2].set_xlabel('Year')
    axs[2].grid(True)

    plt.show()

    print(leh_df[variable])

    return a


def indus_map():
    """ Returns a map of the Indus river """

    rivers = cf.NaturalEarthFeature(category='physical', name='rivers_lake_centerlines', scale='50m', facecolor='none', edgecolor='lightblue')

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([55, 95, 15, 45])
    ax.add_feature(cf.BORDERS)
    ax.add_feature(rivers, linewidth=1)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.text(64, 27, 'PAKISTAN')
    ax.text(63, 33, 'AFGHANISTAN')
    ax.text(80, 36, 'CHINA')
    ax.text(74, 28, 'INDIA')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title('Indus River \n \n')
    plt.show()
