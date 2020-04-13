# Precipitation Data Exploration

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
import cartopy.feature as cf

# open data

tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
mpl_filepath = '/Users/kenzatazi/Downloads/era5_msl_monthly_1979-2019.nc.download'

tp= xr.open_dataset(tp_filepath)
tp_da = tp.tp


# Create a feature for UIB

"""
karakoram = cf.NaturalEarthFeature(
        category='physical',
        name='geography_regions_polys',
        scale='110m',
        facecolor='red')
"""
UIB = tp_da.sel(latitude=slice(37.0, 28.0), longitude=slice(71.0, 82.0))


# Map of annual average for 1979

UIB_1979 = UIB.sel(time=slice('1979-01-16T12:00:00', '1980-01-16T12:00:00'))
UIB_1979_sum = UIB_1979.sum(dim='time')
vmax = UIB_1979_sum.max()
vmin = UIB_1979_sum.min()

plt.figure()
ax = plt.subplot(projection=ccrs.PlateCarree())
ax.set_extent([55, 95, 15, 45])
UIB_1979_sum.plot(cmap='magma_r',  cbar_kwargs={'label': '\n Total precipitation [m]'})
ax.add_feature(cf.BORDERS)
ax.coastlines()
ax.gridlines(draw_labels=True)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.title('Upper Indus Basin Total Precipitation (1979) \n \n')
plt.show()


# Maps of average annual change from 1979 to 1989, 1999, 2009 and 2019

UIB_1989 = UIB.sel(time=slice('1989-01-16T12:00:00', '1990-01-16T12:00:00'))
UIB_1989_sum = UIB_1989.sum(dim='time')
UIB_1989_change = UIB_1989_sum/ UIB_1979_sum

UIB_1999 = UIB.sel(time=slice('1999-01-16T12:00:00', '2000-01-16T12:00:00'))
UIB_1999_sum = UIB_1999.sum(dim='time')
UIB_1999_change = UIB_1999_sum/ UIB_1979_sum

UIB_2009 = UIB.sel(time=slice('2009-01-16T12:00:00', '2010-01-16T12:00:00'))
UIB_2009_sum = UIB_2009.sum(dim='time')
UIB_2009_change = UIB_2009_sum/ UIB_1979_sum

UIB_2019 = UIB.sel(time=slice('2019-01-16T12:00:00', '2020-01-16T12:00:00'))
UIB_2019_sum = UIB_2019.sum(dim='time')
UIB_2019_change = UIB_2019_sum/ UIB_1979_sum

fig, ax = plt.subplots(2, 2)
UIB_1989_change.plot(cmap='magma_r', ax=ax[0,0], cbar_kwargs={'label': '\n Total precipitation [m]'})
UIB_1999_change.plot(cmap='magma_r', ax=ax[0,1], cbar_kwargs={'label': '\n Total precipitation [m]'})
UIB_2009_change.plot(cmap='magma_r', ax=ax[1,0], cbar_kwargs={'label': '\n Total precipitation [m]'})
UIB_2019_change.plot(cmap='magma_r', ax=ax[1,1], cbar_kwargs={'label': '\n Total precipitation [m]'})



# Time series for Gilgit, Skardu and Leh

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




