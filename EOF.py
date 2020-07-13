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

print('opening file')
z200_da = xr.open_dataset(z200_filepath)

print('dropping nans')
z200 = z200_da.sel(expver=1).drop('expver').dropna(dim='time')

grouped_da = z200.resample(time="1MS").mean(dim="time")

EOF_ds_list = []

for y in range(36):

    for m in tqdm(np.arange(1,13)):
        
        ## Select subperiod
        
        if m < 9:
            start_date = str(1983+y) + '-0' + str(m) + '-01T12:00:00'
            end_date = str(1983+y+1) + '-0' +  str(m+1) +'-01T12:00:00'
        
        if m == 9:
            start_date = str(1983+y) + '-0' + str(m) + '-01T12:00:00'
            end_date = str(1983+y+1) + '-' +  str(m+1) +'-01T12:00:00'
        
        if m > 9:
            start_date = str(1983+y) + '-' + str(m) + '-01T12:00:00'
            end_date = str(1983+y) + '-' + str(m+1) +'-01T12:00:00'
        
        if m == 12:
            start_date = str(1983+y) + '-' + str(m) + '-01T12:00:00'
            end_date = str(1983+y+1) + '-' +  str(1) +'-01T12:00:00'

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
        EOF = EOFs.reshape(1, 721, 1440) * 100
        ds = xr.Dataset({'EOF': (('time','latitude','longitude'), EOF)}, 
                        coords={'time':[start_date], 'latitude':z200.latitude, 'longitude': z200.longitude})
        EOF_ds_list.append(ds)

EOF2 = xr.combine_by_coords(datasets=EOF_ds_list)
EOF2.to_netcdf(path='/gws/nopw/j04/bas_climate/users/ktazi/z200_EOF2.nc')


'''    
plt.figure()
ax = plt.subplot(projection=ccrs.PlateCarree())
g = da.plot(cbar_kwargs={'label': '\n EOF2', 'extend':'neither', 'pad':0.10})
ax.add_feature(cf.BORDERS)
ax.coastlines()
ax.gridlines(draw_labels=True)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()
'''