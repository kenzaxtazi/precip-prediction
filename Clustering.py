
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import PrecipitationDataExploration as pde
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
import cartopy.feature as cf
import pandas as pd
import seaborn as sns


tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'
dem_filepath = '/Users/kenzatazi/Downloads/elev.0.25-deg.nc'


dem = xr.open_dataset(dem_filepath)
dem_da = (dem.data).sum(dim='time')
sliced_dem = dem_da.sel(lat=slice(38, 30), lon=slice(71.25, 82.75)) 


da = pde.apply_mask(tp_filepath, mask_filepath)
UIB_cum = pde.cumulative_montly(da)*1000
decades = [1980, 1990, 2000, 2010]
N = np.arange(2, 11, 1)

JFM = UIB_cum.where(UIB_cum.time.dt.month <= 3)
OND = UIB_cum.where(UIB_cum.time.dt.month >= 10)
AMJ = UIB_cum.where(UIB_cum.time.dt.month < 7).where(UIB_cum.time.dt.month > 3)
JAS = UIB_cum.where(UIB_cum.time.dt.month < 10).where(UIB_cum.time.dt.month > 6)

seasons = [JFM, AMJ, JAS, OND]

scmap = {0:'Reds', 1:'Oranges', 2:'Blues', 3:'Greens'}


for s in range(4):

    UIB_cluster_list = []

    for n in N:

        UIB_dec_list = []

        for d in decades: 

            # Median cumulative rainfall for decades
            UIB_d = seasons[s].sel(time=slice(str(d)+'-01-01T12:00:00', str(d+10)+'-01-01T12:00:00'))
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


    UIB_clusters1 = xr.concat(UIB_cluster_list[0:3], pd.Index(np.arange(2,5), name='N'))
    UIB_clusters2 = xr.concat(UIB_cluster_list[3:6], pd.Index(np.arange(5,8), name='N'))
    UIB_clusters3 = xr.concat(UIB_cluster_list[6:9], pd.Index(np.arange(8,11), name='N'))

    
    g= UIB_clusters1.plot(x='longitude', y='latitude', col='Decade', row='N', vmin=-2, vmax=10,
                        subplot_kws={'projection': ccrs.PlateCarree()}, cmap=scmap[s],
                        add_colorbar=False)
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x='lon', y='lat', ax=ax, cmap='bone_r')


    g= UIB_clusters2.plot(x='longitude', y='latitude', col='Decade', row='N', vmin=-2, vmax=10,
                        subplot_kws={'projection': ccrs.PlateCarree()}, cmap=scmap[s],
                        add_colorbar=False)
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x='lon', y='lat', ax=ax, cmap='bone_r')
        

    g= UIB_clusters3.plot(x='longitude', y='latitude', col='Decade', row='N', vmin=-2, vmax=10,
                        subplot_kws={'projection': ccrs.PlateCarree()}, cmap=scmap[s],
                        add_colorbar=False)
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x='lon', y='lat', ax=ax, cmap='bone_r')

plt.show()

