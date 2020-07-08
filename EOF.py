# EOF

import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.cm as cm
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import FileDownloader as fd

from sklearn.decomposition import PCA
from tqdm import tqdm

## Load the data
z200_filepath = fd.update_cds_hourly_data(variables=['geopotential'], pressure_level='200', path='/gws/nopw/j04/bas_climate/users/ktazi', qualifier='global_z200')
z200 = xr.open_dataset(z200_filepath).dropna(dim='time')

EOF_da_list = []

for y in tqdm(range(40)):

    for m in np.arange(1,13):
        
        ## Select subperiod
        
        if m < 9:
            start_date = str(1981+y) + '-0' + str(m) + '-01T12:00:00'
            end_date = str(1981+y+1) + '-0' +  str(m+1) +'-01T12:00:00'
        
        if m == 9:
            start_date = str(1981+y) + '-0' + str(m) + '-01T12:00:00'
            end_date = str(1981+y+1) + '-' +  str(m+1) +'-01T12:00:00'
        
        if m > 9:
            start_date = str(1981+y) + '-' + str(m) + '-01T12:00:00'
            end_date = str(1981+y) + '-' + str(m+1) +'-01T12:00:00'
        
        if m == 12:
            start_date = str(1981+y) + '-' + str(m) + '-01T12:00:00'
            end_date = str(1981+y+1) + '-' +  str(1) +'-01T12:00:00'

        z200_month = z200.sel(time=slice(start_date, end_date))

        ## Reshape in 2D time space
        arr = z200_month.z.values
        X = arr.reshape(len(z200_month.time), -1)  

        # Fit PCA
        skpca = PCA() #instantiates PCA object
        skpca.fit(X) # fit
        
        '''
        ### Save fitted PCA oject
        joblib.dump(skpca, '../EOF.pkl', compress=9)

        ### Plot of PCAs
        f, ax = plt.subplots(figsize=(5,5))
        ax.plot(skpca.explained_variance_ratio_[0:10]*100)
        ax.plot(skpca.explained_variance_ratio_[0:10]*100,'ro')
        ax.set_title("% of variance explained", fontsize=14)
        ax.grid()
        '''

        ### The Empirical Orthogonal Functions (EOFs)
        EOFs = skpca.components_
        EOFs = EOFs[1,:]

        ## 2D field reconstruction
        EOF = EOFs.reshape(721, 1440)
        da = xr.DataArray(data=EOF, coords=[z200.latitude, z200.longitude, start_date], dims=['latitude','longitude', 'time'])
        EOF_da_list.append(da)

EOF2 = xr.DataArray(data=EOF_da_list)
EOF2.to_netcdf(path='/gws/nopw/j04/bas_climate/users/ktazi/z200_EOF2.nc')


'''    
plt.figure()
ax = plt.subplot(projection=ccrs.PlateCarree())
ax.set_extent([0, 150, -60, 60])
g = da.plot(cbar_kwargs={'label': '\n EOF2', 'extend':'neither'})
g.cmap.set_under('white')
ax.add_feature(cf.BORDERS)
ax.coastlines()
ax.gridlines(draw_labels=True)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()
'''