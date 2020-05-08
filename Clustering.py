
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import PrecipitationDataExploration as pde
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
import cartopy.feature as cf


tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'

da = pde.apply_mask(tp_filepath, mask_filepath)

# Cumulative rainfall for 1979
UIB_cum = pde.cumulative_montly(da)
UIB_2000 = UIB_cum.sel(time=slice('2000-01-01T12:00:00', '2001-01-01T12:00:00'))
UIB_2000_sum = UIB_2000.median(dim='time')*1000

# To DataFrame
multi_index_df = UIB_2000_sum.to_dataframe('Precipitation')
df= multi_index_df.reset_index()
df_clean = df[df['Precipitation']>0]
X = (df_clean['Precipitation'].values).reshape(-1, 1)

kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
df_clean['Labels'] = kmeans.labels_

df_plot = df_clean[['Labels', 'latitude', 'longitude']]
df_pv = df_plot.pivot(index='latitude', columns='longitude')
df_pv = df_pv.droplevel(0, axis=1)
da = xr.DataArray(data=df_pv)

plt.figure()
ax = plt.subplot(projection=ccrs.PlateCarree())
ax.set_extent([70, 83, 30, 38])

a= da.plot(cmap='magma_r', vmin=1, cbar_kwargs={'label': '\n Sub basins', 'extend':'neither'})
ax.add_feature(cf.BORDERS)
ax.coastlines()
ax.gridlines(draw_labels=True)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()