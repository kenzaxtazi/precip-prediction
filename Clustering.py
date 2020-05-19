
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import PrecipitationDataExploration as pde
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
import cartopy.feature as cf
import pandas as pd
import seaborn as sns


# Filepaths 
tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'
dem_filepath = '/Users/kenzatazi/Downloads/elev.0.25-deg.nc'


# Digital Elevation Model
dem = xr.open_dataset(dem_filepath)
dem_da = (dem.data).sum(dim='time')
sliced_dem = dem_da.sel(lat=slice(38, 30), lon=slice(71.25, 82.75)) 


# Precipitation Data
da = pde.apply_mask(tp_filepath, mask_filepath)
UIB_cum = pde.cumulative_montly(da)*1000


# Decades segmentation
decades = [1980, 1990, 2000, 2010]


# Clusters
N = np.arange(2, 11, 1)


def seasonal_clusters(UIB_cum, sliced_dem, N, decades):
    """ 
    K-means clustering of precipitation data as a function of seasons, decades and number of clusters. 
    Returns spatial graphs, overlayed with the local topography.

    Inputs:
        UIB_cum: precipitation data (data array)
        sliced_dem: local topography data (data array)
        N: number of cluster to perform kmeans on (list)
        decades: decades to analyse (list )

    Returns:
        FacetGrid plots
    """
    
    # Seperation into seasons

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
                
                # K-means
                kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
                df_clean['Labels'] = kmeans.labels_

                # To DataArray
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


def annual_clusters(UIB_cum, sliced_dem, N, decades):
    """ 
    K-means clustering of annual precipitation data as a function of decades and number of clusters. 
    Returns spatial graphs, overlayed with the local topography.

    Inputs:
        UIB_cum: precipitation data (data array)
        sliced_dem: local topography data (data array)
        N: number of cluster to perform kmeans on (list)
        decades: decades to analyse (list )

    Returns:
        FacetGrid plots
    """

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
            
            # K-means
            kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
            df_clean['Labels'] = kmeans.labels_

            # To DataArray
            df_plot = df_clean[['Labels', 'latitude', 'longitude']]
            df_pv = df_plot.pivot(index='latitude', columns='longitude')
            df_pv = df_pv.droplevel(0, axis=1)
            da = xr.DataArray(data=df_pv, name=str(d)+'s')

            UIB_dec_list.append(da)
        
        UIB_cluster_list.append(xr.concat(UIB_dec_list, pd.Index(['1980s', '1990s', '2000s', '2010s'], name='Decade')))

    UIB_clusters1 = xr.concat(UIB_cluster_list[0:3], pd.Index(np.arange(2,5), name='N'))
    UIB_clusters2 = xr.concat(UIB_cluster_list[3:6], pd.Index(np.arange(5,8), name='N'))
    UIB_clusters3 = xr.concat(UIB_cluster_list[6:9], pd.Index(np.arange(8,11), name='N'))

    g= UIB_clusters1.plot(x='longitude', y='latitude', col='Decade', row='N', add_colorbar=False,
                          subplot_kws={'projection': ccrs.PlateCarree()})
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x='lon', y='lat', ax=ax, cmap='bone_r')

    g= UIB_clusters2.plot(x='longitude', y='latitude', col='Decade', row='N', add_colorbar=False,
                          subplot_kws={'projection': ccrs.PlateCarree()})
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x='lon', y='lat', ax=ax, cmap='bone_r')   

    g= UIB_clusters3.plot(x='longitude', y='latitude', col='Decade', row='N', add_colorbar=False,
                          subplot_kws={'projection': ccrs.PlateCarree()})
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x='lon', y='lat', ax=ax, cmap='bone_r')

    plt.show()


def timeseries_clusters(UIB_cum, sliced_dem, N, decades, filter=False):
    """ 
    K-means clustering of precipitation timeseries as a function of decades and number of clusters. 
    Returns spatial graphs, overlayed with the local topography.

    Inputs:
        UIB_cum: precipitation data (data array)
        sliced_dem: local topography data (data array)
        N: number of cluster to perform kmeans on (list)
        decades: decades to analyse (list )

    Returns:
        FacetGrid plots
    """
    UIB_cluster_list = []

    for n in N:

        UIB_dec_list = []

        for d in decades: 

            # Median cumulative rainfall for decades
            UIB_d = UIB_cum.sel(time=slice(str(d)+'-01-01T12:00:00', str(d+10)+'-01-01T12:00:00'))

            # To DataFrame
            multi_index_df = UIB_d.to_dataframe('Precipitation')
            df= multi_index_df.reset_index()
            df_clean = df[df['Precipitation']>0]
            table = pd.pivot_table(df_clean, values='Precipitation', index=['latitude', 'longitude'],
                               columns=['time'])
            X = table.interpolate()
    
            # K-means
            kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
            
            if filter == False:
                X['Labels'] = kmeans.labels_
                filtered_df = X

            if filter == True:
                filtered_df = filtering(X, kmeans)
            
            # To DataArray
            df_plot = filtered_df.reset_index()[['Labels', 'latitude', 'longitude']]
            df_pv = df_plot.pivot(index='latitude', columns='longitude')
            df_pv = df_pv.droplevel(0, axis=1)
            da = xr.DataArray(data=df_pv, name=str(d)+'s')

            UIB_dec_list.append(da)
        
        UIB_cluster_list.append(xr.concat(UIB_dec_list, pd.Index(['1980s', '1990s', '2000s', '2010s'], name='Decade')))

    UIB_clusters1 = xr.concat(UIB_cluster_list[0:3], pd.Index(np.arange(2,5), name='N'))
    UIB_clusters2 = xr.concat(UIB_cluster_list[3:6], pd.Index(np.arange(5,8), name='N'))
    UIB_clusters3 = xr.concat(UIB_cluster_list[6:9], pd.Index(np.arange(8,11), name='N'))
    
    g= UIB_clusters1.plot(x='longitude', y='latitude', col='Decade', row='N', add_colorbar=False,
                          subplot_kws={'projection': ccrs.PlateCarree()})
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x='lon', y='lat', ax=ax, cmap='bone_r')

    g= UIB_clusters2.plot(x='longitude', y='latitude', col='Decade', row='N', add_colorbar=False,
                          subplot_kws={'projection': ccrs.PlateCarree()})
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x='lon', y='lat', ax=ax, cmap='bone_r')        

    g= UIB_clusters3.plot(x='longitude', y='latitude', col='Decade', row='N', add_colorbar=False,
                          subplot_kws={'projection': ccrs.PlateCarree()})
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x='lon', y='lat', ax=ax, cmap='bone_r')

    plt.show()



def soft_clustering_weights(data, cluster_centres, **kwargs):
    
    """
    Taken from: https://towardsdatascience.com/confidence-in-k-means-d7d3a13ca856 

    Function to calculate the weights from soft k-means
    data: Array of data. shape = N x F, for N data points and F Features
    cluster_centres: Array of cluster centres. shape = Nc x F, for Nc number of clusters. Input kmeans.cluster_centres_ directly.
    param: m - keyword argument, fuzziness of the clustering. Default 2
    """
    
    # Fuzziness parameter m>=1. Where m=1 => hard segmentation
    m = 2
    if 'm' in kwargs:
        m = kwargs['m']
    
    Nclusters = cluster_centres.shape[0]
    Ndp = data.shape[0]
    Nfeatures = data.shape[1]

    # Get distances from the cluster centres for each data point and each cluster
    EuclidDist = np.zeros((Ndp, Nclusters))
    for i in range(Nclusters):
        EuclidDist[:,i] = np.sum((data-np.tile(cluster_centres[i], (Ndp, 1)))**2,axis=1)
        
    # Denominator of the weight from wikipedia:
    invWeight = EuclidDist**(2/(m-1)) * np.tile(np.sum((1./EuclidDist)**(2/(m-1)),axis=1).reshape(-1,1), (1, Nclusters))
    Weight = 1./invWeight
    
    return Weight


def filtering (df, kmeans, thresh=0.7):
    weights =  soft_clustering_weights(df, kmeans.cluster_centers_)
    max_weights = np.amax(weights, axis=1)
    df['weights'] = max_weights
    df['Labels'] = kmeans.labels_
    filtered_df = df[df['weights']> thresh]
    return filtered_df


