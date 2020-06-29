# Clustering

import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf

from sklearn.cluster import KMeans

import DataExploration as de
import DataPreparation as dp


if name in if __name__ == "__main__":

    # Filepaths 
    tp_filepath = 'Data/era5_tp_monthly_1979-2019.nc'
    mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'
    dem_filepath = 'Data/elev.0.25-deg.nc'


    # Digital Elevation Model
    dem = xr.open_dataset(dem_filepath)
    dem_da = (dem.data).sum(dim='time')
    sliced_dem = dem_da.sel(lat=slice(38, 30), lon=slice(71.25, 82.75)) 


    # Precipitation Data
    da = dp.apply_mask(tp_filepath, mask_filepath)
    UIB_cum = dp.cumulative_monthly(da.tp)*1000


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


def gp_clusters(tp_da, N=3, filter=0.7, plot=False , confidence_plot=False):

    """ 
    Returns cluster masks for data separation.
    
    Inputs:
        tp_da: total precipitation data array
        N: number of clusters
        filter: soft k-means filtering threshold, float btw 0 (no filtering) and 1
        plot: boolean for plotting
        confidence_plot: boolean for plotting the k-means weights
    
    Returns:
        List of cluster masks as data arrays
        Cluster plot (optional)
        Confidence plot (optional)
    """
    names = {0:'Gilgit', 1:'Ngari', 2:'Khyber'}

    multi_index_df = tp_da.to_dataframe()
    df= multi_index_df.reset_index()
    df_clean = df[df['tp']>0]
    table = pd.pivot_table(df_clean, values='tp', index=['latitude', 'longitude'], columns=['time'])
    X = table.interpolate()
    
    # Soft k-means
    kmeans = KMeans(n_clusters=N, random_state=0).fit(X)
    filtered_df = filtering(X, kmeans, thresh=filter)

    # Seperate labels and append as DataArray
    reset_df = filtered_df.reset_index()[['Labels', 'latitude', 'longitude']]
    clusters = []

    for i in range(N):
        cluster_df = reset_df[reset_df['Labels']== i]
        df_pv = cluster_df.pivot(index='latitude', columns='longitude')
        df_pv = df_pv.droplevel(0, axis=1)
        cluster_da = xr.DataArray(data=df_pv, name=str(i))
        cluster_da.to_netcdf(path= names[i]+'_mask.nc')
        clusters.append(cluster_da)

    if plot == True:
        
        df_pv = reset_df.pivot(index='latitude', columns='longitude')
        df_pv = df_pv.droplevel(0, axis=1)
        da = xr.DataArray(data=df_pv, name='\n Clusters')

        #Plot
        plt.figure()
        ax = plt.subplot(projection=ccrs.PlateCarree())
        ax.set_extent([71, 83, 30, 38])
        c = ['#2460A7FF', '#85B3D1FF', '#D9B48FFF']
        g= da.plot(x='longitude', y='latitude', add_colorbar=False, ax=ax, levels=N+1) # cbar_kwargs={'ticks':[0.4, 1.2, 2], 'pad':0.10})
        cbar = plt.colorbar(g, ticks=[0.4, 1.2, 2], pad=0.10)
        cbar.ax.set_yticklabels(['Gilgit regime', 'Ngari regime', 'Khyber regime'])
        ax.gridlines(draw_labels=True)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.show()

    if confidence_plot == True:
        df_clean = df[df['tp']>0]
        table = pd.pivot_table(df_clean, values='tp', index=['latitude', 'longitude'], columns=['time'])
        X = table.interpolate()
        # Soft k-means
        kmeans = KMeans(n_clusters=N, random_state=0).fit(X)
        unfiltered_df = filtering(X, kmeans, thresh=0)

        conf_df = unfiltered_df.reset_index()[['weights', 'latitude', 'longitude']]
        df_pv = conf_df.pivot(index='latitude', columns='longitude')
        df_pv = df_pv.droplevel(0, axis=1)
        conf_da = xr.DataArray(data=df_pv, name='Confidence')

        #Plot
        plt.figure()
        ax = plt.subplot(projection=ccrs.PlateCarree())
        ax.set_extent([71, 83, 30, 38])
        g= conf_da.plot(x='longitude', y='latitude', add_colorbar=True, ax=ax,
                        vmin=0, vmax=1, cmap='magma', cbar_kwargs={'pad':0.10})
        ax.gridlines(draw_labels=True)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.show()
        
    return clusters


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


