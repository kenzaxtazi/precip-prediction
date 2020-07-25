# Data Exploration

import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.cm as cm
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from shapely.geometry import Polygon, shape, LinearRing
from cartopy.io import shapereader
from cartopy import config
from scipy import signal

import FileDownloader as fd
import DataPreparation as dp
import DataDownloader as dd

data_filepath = fd.update_cds_monthly_data()
mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'

'''
tp_filepath = 'Data/era5_tp_monthly_1979-2019.nc'
mpl_filepath = 'Data/era5_msl_monthly_1979-2019.nc'
'''


def annual_map(data_filepath, mask_filepath, variable, year, cumulative=False):
    """ Map of cumulative precipitation for 1979 """

    da = dd.apply_mask(data_filepath, mask_filepath)
    da_var = da[variable]
    da_year = da_var.sel(time=slice(str(year)+'-01-16T12:00:00', str(year+1)+'-01-01T12:00:00'))

    if cumulative is True:
        da_processed= dp.cumulative_monthly(da_year)
        da_final = da_processed.sum(dim='time')
    else:
        da_final = da_year.mean(dim='time') # TODO weighted mean
   

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    g = da_final.plot(cmap='magma_r', vmin=0.01, cbar_kwargs={'label': '\n Total precipitation [m]', 'extend':'neither'})
    g.cmap.set_under('white')
    ax.add_feature(cf.BORDERS)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title('Upper Indus Basin Total Precipitation ' + str(year) + '\n \n')
    plt.show()


def change_maps(variable, year): #TODO fix this 
    """ Maps of average annual change from 1979 to 1989, 1999, 2009 and 2019 """

    da = xr.open_dataset(data_filepath)

    if 'expver' in list(da.dims):
        print('expver found')
        da = da.sel(expver=1)

    da_var = da[variable]
    da_1979 = da_var.sel(time=slice(str(year)+'-01-16T12:00:00', str(year+1)+'-01-01T12:00:00'))
    da_processed= dp.cumulative_monthly(da_1979)
    UIB_1979_sum = da_processed.sum(dim='time')

    UIB_1989 = da_var.sel(time=slice('1989-01-01T12:00:00', '1990-01-01T12:00:00'))
    UIB_1989_sum = dp.cumulative_monthly(UIB_1989).sum(dim='time')
    UIB_1989_change = UIB_1989_sum/ UIB_1979_sum - 1 

    UIB_1999 = da_var.sel(time=slice('1999-01-01T12:00:00', '2000-01-01T12:00:00'))
    UIB_1999_sum = dp.cumulative_monthly(UIB_1999).sum(dim='time')
    UIB_1999_change = UIB_1999_sum/ UIB_1979_sum - 1

    UIB_2009 = da_var.sel(time=slice('2009-01-01T12:00:00', '2010-01-01T12:00:00'))
    UIB_2009_sum = dp.cumulative_monthly(UIB_2009).sum(dim='time')
    UIB_2009_change = UIB_2009_sum/ UIB_1979_sum - 1

    UIB_2019 = da_var.sel(time=slice('2019-01-01T12:00:00', '2020-01-01T12:00:00'))
    UIB_2019_sum = dp.cumulative_monthly(UIB_2019).sum(dim='time')
    UIB_2019_change = UIB_2019_sum/ UIB_1979_sum - 1

    UIB_changes = xr.concat([UIB_1989_change, UIB_1999_change, UIB_2009_change, UIB_2019_change],
                            pd.Index(['1989', '1999', '2009', '2019'], name='year'))

    g= UIB_changes.plot(x='longitude', y='latitude', col='year', col_wrap=2,
                        subplot_kws={'projection': ccrs.PlateCarree()},
                        cbar_kwargs={'label': 'Precipitation change', 'format': tck.PercentFormatter(xmax=1.0)})

    for ax in g.axes.flat:
        ax.coastlines()
        ax.gridlines()
        ax.set_extent([71, 83, 30, 38])
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


def averaged_timeseries(mask_filepath, variable='tp', longname='Total precipitation [m/day]'):
    """ Timeseries for the Upper Indus Basin"""

    df = dd.download_data(mask_filepath)

    df_var = df[['time',variable]]

    df_var['time'] = df_var['time'].astype(np.datetime64)
    df_mean = df_var.groupby('time').mean()

    df_mean.plot()
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

def detrend(df, axis):
    detrended_array = signal.detrend(data=df, axis=axis)
    df.update(detrended_array)
    return df

def spatial_autocorr(variable, mask_filepath): # TODO
    """ Plots spatial autocorrelation """
    
    df = dd.download_data(mask_filepath)
    # detrend
    table = pd.pivot_table(df, values='tp', index=['latitude', 'longitude'], columns=['time'])
    trans_table = table.T
    detrended_table = detrend(trans_table, axis=0)
    corr_table = detrended_table.corr()
    print(corr_table)

    corr_khyber = corr_table.loc[(34.5,73.0)]
    corr_gilgit = corr_table.loc[(36.0,75.0)]
    corr_ngari = corr_table.loc[(33.5,79.0)]

    corr_list = [corr_khyber, corr_gilgit, corr_ngari]

    for corr in corr_list:

        df_reset = corr.reset_index().droplevel(1, axis=1)
        df_pv = df_reset.pivot(index='latitude', columns='longitude')
        df_pv = df_pv.droplevel(0, axis=1)
        da = xr.DataArray(data= df_pv, name='Correlation')

        #Plot

        plt.figure()
        ax = plt.subplot(projection=ccrs.PlateCarree())
        ax.set_extent([71, 83, 30, 38])
        g= da.plot(x='longitude', y='latitude', add_colorbar=True, ax=ax,
                        vmin=-1, vmax=1, cmap='coolwarm', cbar_kwargs={'pad':0.10})
        ax.gridlines(draw_labels=True)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    plt.show()


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


def indus_map(mask_filepath):
    """ Returns two maps: global map with box, zoomed in map of the Indus river """
    
    # River shapefiles
    fpath = '/Users/kenzatazi/Downloads/ne_50m_rivers_lake_centerlines_scale_rank/ne_50m_rivers_lake_centerlines_scale_rank.shp'
    as_shp = shapereader.Reader(fpath)

    # UIB shapefile
    uib_path = '/Users/kenzatazi/Downloads/UpperIndus_HP_shapefile/UpperIndus_HP.shp'
    uib_shape = shapereader.Reader(uib_path)

    # Other phyical features
    land_50m = cf.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor=cf.COLORS['land'])
    ocean_50m = cf.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='face', facecolor=cf.COLORS['water'])

    # Regional rectangle
    nvert = 100
    lonmin, lonmax, latmin, latmax = [60, 80, 20, 40]
    lons = np.r_[np.linspace(lonmin, lonmin, nvert), np.linspace(lonmin, lonmax, nvert), np.linspace(lonmax, lonmax, nvert)].tolist()
    lats = np.r_[np.linspace(latmin, latmax, nvert), np.linspace(latmax, latmax, nvert), np.linspace(latmax, latmin, nvert)].tolist()
    pgon = LinearRing(list(zip(lons, lats)))
    
    # Projection for global map 
    glb_proj = ccrs.NearsidePerspective(central_longitude=70, central_latitude=30)

    plt.figure('Global')
    ax = plt.subplot(projection=glb_proj)
    ax.add_feature(ocean_50m)
    ax.add_feature(land_50m)
    ax.coastlines('50m', linewidth=0.4)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
    ax.add_geometries([pgon], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red')

    plt.figure('Zoomed in')
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([60, 85, 20, 40])

    ax.add_feature(land_50m)
    ax.add_feature(ocean_50m)
    ax.add_feature(cf.BORDERS.with_scale('50m'), linewidth=0.5)
    ax.coastlines('50m', linewidth=0.5)
    
    for rec in uib_shape.records():
        ax.add_geometries([rec.geometry], ccrs.AlbersEqualArea(central_longitude=125, central_latitude=-15, standard_parallels=(7,-32)),
                          edgecolor='red', facecolor='red', alpha=0.1)

    # ax.add_feature(rivers)
    for rec in as_shp.records():
        if rec.attributes['name'] == 'Indus':
            ax.add_geometries([rec.geometry], ccrs.PlateCarree(), edgecolor='steelblue', facecolor='None')
        pass


    ax.text(64, 28, 'PAKISTAN', c='gray')
    ax.text(62.5, 33.5, 'AFGHANISTAN', c='gray')
    ax.text(80, 37, 'CHINA', c='gray')
    ax.text(77, 25, 'INDIA', c='gray')
    ax.text(62, 22, 'Arabian Sea', c='navy', alpha=0.5)
    ax.text(70.3, 32, 'Indus river', c='steelblue', rotation=55, size=7)
    
    ax.set_xticks([60, 65, 70, 75, 80, 85])
    ax.set_xticklabels(['60°E', '65°E', '70°E', '75°E', '80°E', '85°E'])

    ax.set_yticks([20, 25, 30, 35, 40])
    ax.set_yticklabels(['20°N', '25°N', '30°N', '35°N', '40°N'])

    plt.title('Indus River \n \n')
    plt.show()

def telcon_map():
    # UIB shapefile
    uib_path = '/Users/kenzatazi/Downloads/UpperIndus_HP_shapefile/UpperIndus_HP.shp'
    uib_shape = shapereader.Reader(uib_path)

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree(central_longitude=70))
    ax.coastlines('50m', linewidth=0.5)

    for rec in uib_shape.records():
        ax.add_geometries([rec.geometry], ccrs.AlbersEqualArea(central_longitude=125, central_latitude=-15, standard_parallels=(7,-32)),
                        edgecolor='red', facecolor='red', alpha=0.1)
    
    plt.show()


def tp_vs(mask_filepath, variable, cluster_mask=None, longname=''):

    '''
    df = dp.download_data(mask_filepath)
    df_var = df[['time','tp', variable]]
    #df_mean = df_var.groupby('time').mean()
    '''
    
    if cluster_mask == None:
        da = dd.download_data(mask_filepath)
    else:
        cds_filepath = fd.update_cds_monthly_data()
        da = dd.apply_mask(cds_filepath, cluster_mask)
        df = da.to_dataframe().reset_index()

    df = df[['time','tp', variable]]
    #gilgit = ds.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    df_var = df.dropna()

    # Plot    
    df_var.plot.scatter(x=variable, y='tp', alpha=.2, c='b')
    plt.title('Upper Indus Basin')
    plt.ylabel('Total precipitation [m/day]')
    plt.xlabel(longname)
    plt.grid(True)
    plt.show()