
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import PrecipitationDataExploration as pde
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
import cartopy.feature as cf
import pandas as pd


tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'

da = pde.apply_mask(tp_filepath, mask_filepath)
UIB_cum = pde.cumulative_montly(da)*1000

decades = [1980, 1990, 2000, 2010]
N = np.arange(2, 6, 1)

index_list = []
UIB_cluster_list = []

for n in N:

    UIB_dec_list = []

    for d in decades: 

        # Median cumulative rainfall for decades
        UIB_d = UIB_cum.sel(time=slice(str(d)+'-01-01T12:00:00', str(d+10)+'-01-01T12:00:00'))
        UIB_median = UIB_d.median(dim='time')

        # To DataFrame
        multi_index_df = UIB_median.to_dataframe('Precipitation')
        df= multi_index_df.reset_index()
        df_clean = df[df['Precipitation']>0]
        X = (df_clean['Precipitation'].values).reshape(-1, 1)

        kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
        df_clean['Labels'] = kmeans.labels_

        df_plot = df_clean[['Labels', 'latitude', 'longitude']]
        df_pv = df_plot.pivot(index='latitude', columns='longitude')
        df_pv = df_pv.droplevel(0, axis=1)
        da = xr.DataArray(data=df_pv, name=str(d)+'s')

        UIB_dec_list.append(da)
    
    UIB_cluster_list.append(xr.concat(UIB_dec_list, pd.Index(['1980s', '1990s', '2000s', '2010s'], name='Decade')))


UIB_clusters = xr.concat(UIB_cluster_list, pd.Index(['10', '11', '12', '13'], name='N'))

g= UIB_clusters.plot(x='longitude', y='latitude', col='Decade', row='N',
                     subplot_kws={'projection': ccrs.PlateCarree()}, cmap='magma_r',
                     add_colorbar=False)

for ax in g.axes.flat:
    ax.coastlines()
    ax.gridlines()
    ax.set_extent([70, 83, 30, 38])
    ax.add_feature(cf.BORDERS)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()