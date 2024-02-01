# Clustering
import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/precip-prediction')  # noqa

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from sklearn.cluster import KMeans

from load import era5, data_dir, location_sel
from maps.plot_data import cumulative_monthly
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker

# Filepaths
mask_filepath = data_dir + "Masks/ERA5_Upper_Indus_mask.nc"
dem_filepath = data_dir + "Elevation/elev.0.25-deg.nc"


# Function inputs

# Digital Elevation Model data
dem = xr.open_dataset(dem_filepath)
dem_da = (dem.data).sum(dim="time")
sliced_dem = dem_da.sel(lat=slice(38, 30), lon=slice(71.25, 82.75))

# Precipitation data
da = era5.download_data('uib', xarray=True)
loc_ds = location_sel.select_basin(da, 'uib')
tp_ds = loc_ds.tp

# Decade list
decades = [1980, 1990, 2000, 2010]

# Cluster list
N = np.arange(2, 11, 1)


def seasonal_clusters(tp_ds, sliced_dem, N, decades):
    """
    K-means clustering of precipitation data as a function of seasons, decades
    and number of clusters. Returns spatial graphs, overlayed with the local
    topography contours.

    Inputs:
        UIB_cum: precipitation data (data array)
        sliced_dem: local topography data (data array)
        N: number of cluster to perform kmeans on (list)
        decades: decades to analyse (list )

    Returns:
        FacetGrid plots
    """

    UIB_cum = cumulative_monthly(tp_ds)

    # Seperation into seasons
    JFM = UIB_cum.where(UIB_cum.time.dt.month <= 3)
    OND = UIB_cum.where(UIB_cum.time.dt.month >= 10)
    AMJ = UIB_cum.where(UIB_cum.time.dt.month < 7).where(
        UIB_cum.time.dt.month > 3)
    JAS = UIB_cum.where(UIB_cum.time.dt.month < 10).where(
        UIB_cum.time.dt.month > 6)

    seasons = [JFM, AMJ, JAS, OND]
    scmap = {0: "Reds", 1: "Oranges", 2: "Blues", 3: "Greens"}

    # Loop over seasons, cluster numbers and decades

    for s in range(4):
        UIB_cluster_list = []

        for n in N:
            UIB_dec_list = []

            for d in decades:
                # Median cumulative rainfall for decades
                UIB_d = seasons[s].sel(
                    time=slice(
                        str(d) + "-01-01T12:00:00", str(d + 10) +
                        "-01-01T12:00:00"
                    )
                )
                UIB_median = UIB_d.median(dim="time")

                # To DataFrame
                multi_index_df = UIB_median.to_dataframe("Precipitation")
                df = multi_index_df.reset_index()
                df_clean = df[df["Precipitation"] > 0]
                X = (df_clean["Precipitation"].values).reshape(-1, 1)

                # K-means
                kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
                df_clean["labels"] = kmeans.labels_

                # To DataArray
                df_plot = df_clean[["labels", "lat", "lon"]]
                df_pv = df_plot.pivot(index="lat", columns="lon")
                df_pv = df_pv.droplevel(0, axis=1)
                da = xr.DataArray(data=df_pv, name=str(d) + "s")

                UIB_dec_list.append(da)

            UIB_cluster_list.append(
                xr.concat(
                    UIB_dec_list,
                    pd.Index(["1980s", "1990s", "2000s",
                             "2010s"], name="Decade"),
                )
            )

        # Combine data for plot formatting
        UIB_clusters1 = xr.concat(
            UIB_cluster_list[0:3], pd.Index(np.arange(2, 5), name="N")
        )
        UIB_clusters2 = xr.concat(
            UIB_cluster_list[3:6], pd.Index(np.arange(5, 8), name="N")
        )
        UIB_clusters3 = xr.concat(
            UIB_cluster_list[6:9], pd.Index(np.arange(8, 11), name="N")
        )

        # Plots
        g = UIB_clusters1.plot(
            x="lon",
            y="lat",
            col="Decade",
            row="N",
            vmin=-2,
            vmax=10,
            subplot_kws={"projection": ccrs.PlateCarree()},
            cmap=scmap[s],
            add_colorbar=False,
        )
        for ax in g.axes.flat:
            ax.set_extent([71, 83, 30, 38])
            sliced_dem.plot.contour(x="lon", y="lat", ax=ax, cmap="bone_r")

        g = UIB_clusters2.plot(
            x="lon",
            y="lat",
            col="Decade",
            row="N",
            vmin=-2,
            vmax=10,
            subplot_kws={"projection": ccrs.PlateCarree()},
            cmap=scmap[s],
            add_colorbar=False,
        )
        for ax in g.axes.flat:
            ax.set_extent([71, 83, 30, 38])
            sliced_dem.plot.contour(x="lon", y="lat", ax=ax, cmap="bone_r")

        g = UIB_clusters3.plot(
            x="lon",
            y="lat",
            col="Decade",
            row="N",
            vmin=-2,
            vmax=10,
            subplot_kws={"projection": ccrs.PlateCarree()},
            cmap=scmap[s],
            add_colorbar=False,
        )
        for ax in g.axes.flat:
            ax.set_extent([71, 83, 30, 38])
            sliced_dem.plot.contour(x="lon", y="lat", ax=ax, cmap="bone_r")

    plt.show()


def annual_clusters(UIB_cum, sliced_dem, N, decades):
    """
    K-means clustering of annual precipitation data as a function of decades
    and number of clusters. Returns spatial graphs, overlayed with the local
    topography.

    Inputs:
        UIB_cum: precipitation data (data array)
        sliced_dem: local topography data (data array)
        N: number of cluster to perform kmeans on (list)
        decades: decades to analyse (list )

    Returns:
        FacetGrid plots
    """
    # Loop over cluster numbers and decades

    UIB_cluster_list = []

    for n in N:
        UIB_dec_list = []

        for d in decades:
            # Median cumulative rainfall for decades
            UIB_d = UIB_cum.sel(
                time=slice(str(d) + "-01-01T12:00:00",
                           str(d + 10) + "-01-01T12:00:00")
            )
            UIB_median = UIB_d.median(dim="time")

            # To DataFrame
            multi_index_df = UIB_median.to_dataframe("Precipitation")
            df = multi_index_df.reset_index()
            df_clean = df[df["Precipitation"] > 0]

            X = (df_clean["Precipitation"].values).reshape(-1, 1)

            # K-means
            kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
            df_clean["labels"] = kmeans.labels_

            # To DataArray
            df_plot = df_clean[["labels", "lat", "lon"]]
            df_pv = df_plot.pivot(index="lat", columns="lon")
            df_pv = df_pv.droplevel(0, axis=1)
            da = xr.DataArray(data=df_pv, name=str(d) + "s")

            UIB_dec_list.append(da)

        UIB_cluster_list.append(
            xr.concat(
                UIB_dec_list,
                pd.Index(["1980s", "1990s", "2000s", "2010s"], name="Decade"),
            )
        )

    # Combine data for plot formatting
    UIB_clusters1 = xr.concat(
        UIB_cluster_list[0:3], pd.Index(np.arange(2, 5), name="N")
    )
    UIB_clusters2 = xr.concat(
        UIB_cluster_list[3:6], pd.Index(np.arange(5, 8), name="N")
    )
    UIB_clusters3 = xr.concat(
        UIB_cluster_list[6:9], pd.Index(np.arange(8, 11), name="N")
    )

    # Plots
    g = UIB_clusters1.plot(
        x="lon",
        y="lat",
        col="Decade",
        row="N",
        add_colorbar=False,
        subplot_kws={"projection": ccrs.PlateCarree()},
    )
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x="lon", y="lat", ax=ax, cmap="bone_r")

    g = UIB_clusters2.plot(
        x="lon",
        y="lat",
        col="Decade",
        row="N",
        add_colorbar=False,
        subplot_kws={"projection": ccrs.PlateCarree()},
    )
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x="lon", y="lat", ax=ax, cmap="bone_r")

    g = UIB_clusters3.plot(
        x="lon",
        y="lat",
        col="Decade",
        row="N",
        add_colorbar=False,
        subplot_kws={"projection": ccrs.PlateCarree()},
    )
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x="lon", y="lat", ax=ax, cmap="bone_r")

    plt.show()


def timeseries_clusters(UIB_cum, sliced_dem, N, decades, filter=0.7):
    """
    Soft k-means clustering of precipitation timeseries as a function of
    decades and number of clusters. Returns spatial graphs, overlayed with
    the local topography.

    Inputs:
        UIB_cum: precipitation data (data array)
        sliced_dem: local topography data (data array)
        N: number of cluster to perform kmeans on (list)
        decades: decades to analyse (list)
        filter: probabilty filtering threshold for soft kmeans
        (float between 0 (no filtering) and 1)

    Returns:
        FacetGrid plots
    """

    # Loop over cluster numbers and decades
    UIB_cluster_list = []

    for n in N:
        UIB_dec_list = []

        for d in decades:
            # Median cumulative rainfall for decades
            UIB_d = UIB_cum.sel(
                time=slice(str(d) + "-01-01T12:00:00",
                           str(d + 10) + "-01-01T12:00:00")
            )

            # To DataFrame
            multi_index_df = UIB_d.to_dataframe("Precipitation")
            df = multi_index_df.reset_index()
            df_clean = df[df["Precipitation"] > 0]
            table = pd.pivot_table(
                df_clean,
                values="Precipitation",
                index=["lat", "lon"],
                columns=["time"],
            )
            X = table.interpolate()

            # K-means
            kmeans = KMeans(n_clusters=n, random_state=0).fit(X)

            if filter == 0:
                X["labels"] = kmeans.labels_
                filtered_df = X

            if filter > 0:
                filtered_df = filtering(X, kmeans, thresh=filter)

            # To DataArray
            df_plot = filtered_df.reset_index(
            )[["labels", "lat", "lon"]]
            df_pv = df_plot.pivot(index="lat", columns="lon")
            df_pv = df_pv.droplevel(0, axis=1)
            da = xr.DataArray(data=df_pv, name=str(d) + "s")

            UIB_dec_list.append(da)

        UIB_cluster_list.append(
            xr.concat(
                UIB_dec_list,
                pd.Index(["1980s", "1990s", "2000s", "2010s"], name="Decade"),
            )
        )

    # Combine data for plot formatting
    UIB_clusters1 = xr.concat(
        UIB_cluster_list[0:3], pd.Index(np.arange(2, 5), name="N")
    )
    UIB_clusters2 = xr.concat(
        UIB_cluster_list[3:6], pd.Index(np.arange(5, 8), name="N")
    )
    UIB_clusters3 = xr.concat(
        UIB_cluster_list[6:9], pd.Index(np.arange(8, 11), name="N")
    )

    # Plots
    g = UIB_clusters1.plot(
        x="lon",
        y="lat",
        col="Decade",
        row="N",
        add_colorbar=False,
        subplot_kws={"projection": ccrs.PlateCarree()},
    )
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x="lon", y="lat", ax=ax, cmap="bone_r")

    g = UIB_clusters2.plot(
        x="lon",
        y="lat",
        col="Decade",
        row="N",
        add_colorbar=False,
        subplot_kws={"projection": ccrs.PlateCarree()},
    )
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x="lon", y="lat", ax=ax, cmap="bone_r")

    g = UIB_clusters3.plot(
        x="lon",
        y="lat",
        col="Decade",
        row="N",
        add_colorbar=False,
        subplot_kws={"projection": ccrs.PlateCarree()},
    )
    for ax in g.axes.flat:
        ax.set_extent([71, 83, 30, 38])
        sliced_dem.plot.contour(x="lon", y="lat", ax=ax, cmap="bone_r")

    plt.show()


def uib_clusters(tp_ds:xr.Dataset, N:int=3, filter:float= None, plot_clusters=False, plot_weights=False) -> list:
    """
    Generate precipiation clusters for UIB.

    Args:
        tp_ds (xr.Dataset): Precipitation data.
        N (int, optional):  Number of clusters. Defaults to 3.
        filter (floatorNone, optional): soft k-means filtering threshold, float between 0 (no
        filtering) and 1. Defaults to None in which case only k-means is performed.
        plot_clusters (bool, optional): Plot (soft) k-means clusters. Defaults to False.
        plot_weights (bool, optional): Plot k-means weights. Defaults to False.

    Returns:
        list: cluster masks as data arrays.
    """
    names = {0: "Khyber", 1: "Ngari", 2: "Gilgit"}

    multi_index_df = tp_ds.to_dataframe()
    df = multi_index_df.reset_index()
    df_clean = df[df["tp"] > 0]
    table = pd.pivot_table(
        df_clean, values="tp", index=["lat", "lon"],
        columns=["time"])
    X = table.interpolate().dropna()

    # Soft k-means
    kmeans = KMeans(n_clusters=N, random_state=0).fit(X)

    if filter != None:
        filtered_df = filtering(X, kmeans, thresh=filter)
        # Seperate labels and append as DataArray
        reset_df = filtered_df.reset_index()[["labels", "lat", "lon"]]
    else:
        table['labels'] = kmeans.predict(X)
        reset_df = table.reset_index()[["labels", "lat", "lon"]]

    ds = reset_df.set_index(["lon", "lat"]).to_xarray()

    clusters = []
    for i in range(N):
        cluster_ds = ds.where(ds.labels == i)
        cluster_ds = cluster_ds.rename({"labels": "overlap"})
        cluster_ds = cluster_ds.where(cluster_ds.overlap !=i, 1)
        clusters.append(cluster_ds)
        if filter != None:
            cluster_ds.to_netcdf(path= data_dir + "Masks/" + names[i] + "_mask_" + str(filter) + ".nc")
        else:
            cluster_ds.to_netcdf(path= data_dir + "Masks/" + names[i] + "_mask.nc")
    
    if plot_clusters is True:
        # Plot
        plt.figure(figsize=(4, 2))
        ax = plt.subplot(projection=ccrs.PlateCarree())
        ax.set_extent([71, 83, 30, 38])
        ds.labels.plot(
            x="lon", y="lat", add_colorbar=False, ax=ax,
            levels=N+1, colors=[ "#D9B48FFF", "#85B3D1FF", "#2460A7FF"])

        # Add legend
        labels = ["Gilgit regime", "Ngari regime", "Khyber regime"]
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#2460A7FF", "#85B3D1FF", "#D9B48FFF"]]
        plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.05, 0.99),
                   frameon=False)   

        ax.set_yticks(np.arange(31, 38), crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(72, 83, 2), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        gl = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree())
        gl.xlocator = mticker.FixedLocator(np.arange(72, 83, 2))
        ax.set_xlabel(" ")  # "Lon")
        ax.set_ylabel(" ")  # ("Lat")
        plt.savefig('regimes_uib_fixed_gridlines.pdf', bbox_inches='tight', dpi=300)
        plt.show()

    if plot_weights is True:
        df_clean = df[df["tp"] > 0]
        table = pd.pivot_table(
            df_clean, values="tp", index=["lat", "lon"],
            columns=["time"])
        X = table.interpolate()

        unfiltered_df = filtering(X, kmeans, thresh=0)
        conf_df = unfiltered_df.reset_index(
        )[["weights", "lat", "lon"]]
        df_pv = conf_df.pivot(index="lat", columns="lon")
        df_pv = df_pv.droplevel(0, axis=1)
        conf_da = xr.DataArray(data=df_pv, name="Confidence")

        # Plot
        plt.figure()
        ax = plt.subplot(projection=ccrs.PlateCarree())
        ax.set_extent([71, 83, 30, 38])
        g = conf_da.plot(
            x="lon",
            y="lat",
            add_colorbar=True,
            ax=ax,
            vmin=0,
            vmax=1,
            cmap="magma",
            cbar_kwargs={"pad": 0.10},
        )
        ax.gridlines(draw_labels=True)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        plt.show()

    return clusters


def soft_clustering_weights(data, cluster_centres, **kwargs):
    """
    Taken from:
    https://towardsdatascience.com/confidence-in-k-means-d7d3a13ca856

    Function to calculate the weights from soft k-means
    data: array of data. shape = N x F, for N data points and
        F Features
    cluster_centres: array of cluster centres. shape = Nc x F, for Nc number
        of clusters. Input kmeans.cluster_centres_ directly.
    param: m - keyword argument, fuzziness of the clustering. Default 2
    """

    # Fuzziness parameter m>=1. Where m=1 => hard segmentation
    m = 2
    if "m" in kwargs:
        m = kwargs["m"]

    Nclusters = cluster_centres.shape[0]
    Ndp = data.shape[0]
    # Nfeatures = data.shape[1]

    # Get distances from the cluster centres for each data point and each
    # cluster
    EuclidDist = np.zeros((Ndp, Nclusters))
    for i in range(Nclusters):
        EuclidDist[:, i] = np.sum(
            (data - np.tile(cluster_centres[i], (Ndp, 1))) ** 2, axis=1
        )

    # Denominator of the weight from wikipedia:
    invWeight = EuclidDist ** (2 / (m - 1)) * np.tile(
        np.sum((1.0 / EuclidDist) ** (2 / (m - 1)), axis=1).reshape(-1, 1),
        (1, Nclusters),
    )
    Weight = 1.0 / invWeight

    return Weight


def filtering(df, kmeans, thresh=0.7):
    weights = soft_clustering_weights(df, kmeans.cluster_centers_)
    max_weights = np.amax(weights, axis=1)
    df["weights"] = max_weights
    df["labels"] = kmeans.labels_
    filtered_df = df[df["weights"] > thresh]
    return filtered_df


def new_gp_clusters(df, N=3, filter=0.7):

    df_clean = df[df["tp"] > 0]
    table = pd.pivot_table(
        df_clean, values="tp", index=["lat", "lon"],
        columns=["time"])
    X = table.interpolate(axis=1)

    # Soft k-means
    kmeans = KMeans(n_clusters=N, random_state=0).fit(X)
    filtered_df = filtering(X, kmeans, thresh=filter)

    # Seperate labels and append as DataArray
    reset_df = filtered_df.reset_index()[["labels", "lat", "lon"]]
    df_combined = pd.merge_ordered(df_clean, reset_df, on=[
                                   "lat", "lon"])
    clean_df = df_combined.dropna()

    return clean_df
