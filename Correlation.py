# Correlation

import datetime
import urllib

import scipy as sp
import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from tqdm import tqdm

import DataExploration as de
import DataDownloader as dd
import FileDownloader as fd
import Clustering as cl


### Filepaths
mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"
eof_filepath = "/gws/nopw/j04/bas_climate/users/ktazi/z200_EOF2.nc"
corr_filepath = "Data/EOF_corr_pval.csv"


def input_correlation_heatmap():

    df = dd.download_data(mask_filepath, all_var=True)

    # create lags
    df["N34-1"] = df["N34"].shift(periods=393)
    df["NAO-1"] = df["NAO"].shift(periods=393)
    df["N4-1"] = df["N4"].shift(periods=393)

    df = df.drop(columns=["time"])
    df_clean = df.dropna()
    df_sorted = df_clean.sort_index(axis=1)
    corr = df_sorted.corr()

    sns.set(style="white")
    plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.triu(
        np.ones_like(corr, dtype=np.bool)
    )  # generate a mask for the upper triangle
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        fmt="0.2f",
        square=True,
        linewidths=0.5,
        annot=True,
        annot_kws={"size": 5},
        cbar_kws={"shrink": 0.5},
    )
    plt.title("Correlation plot for Upper Indus Basin")

    plt.show()


def cluster_correlation_heatmap():

    masks = ["Khyber_mask.nc", "Gilgit_mask.nc", "Ngari_mask.nc"]
    names = ["Gilgit regime", "Ngari regime", "Khyber regime"]

    for i in range(3):
        cluster_df = dd.download_data(masks[i])

        # create lags
        cluster_df["CGTI-1"] = cluster_df["CGTI"].shift(periods=1)
        cluster_df["CGTI-2"] = cluster_df["CGTI"].shift(periods=2)
        cluster_df["CGTI-3"] = cluster_df["CGTI"].shift(periods=3)
        cluster_df["CGTI-4"] = cluster_df["CGTI"].shift(periods=4)
        cluster_df["CGTI-5"] = cluster_df["CGTI"].shift(periods=5)
        cluster_df["CGTI-6"] = cluster_df["CGTI"].shift(periods=6)
        """'
        df_combined['N34-1'] = df_combined['N34'].shift(periods=1)
        df_combined['N34-2'] = df_combined['N34'].shift(periods=2)
        df_combined['N34-3'] = df_combined['N34'].shift(periods=3)
        df_combined['N34-4'] = df_combined['N34'].shift(periods=4)
        df_combined['N34-5'] = df_combined['N34'].shift(periods=5)
        df_combined['N34-6'] = df_combined['N34'].shift(periods=6)

        df_combined['N4-1'] = df_combined['N4'].shift(periods=1)
        df_combined['N4-2'] = df_combined['N4'].shift(periods=2)
        df_combined['N4-3'] = df_combined['N4'].shift(periods=3)
        df_combined['N4-4'] = df_combined['N4'].shift(periods=4)
        df_combined['N4-5'] = df_combined['N4'].shift(periods=5)
        df_combined['N4-6'] = df_combined['N4'].shift(periods=6)
        """
        df = cluster_df(columns=["expver", "time"])
        df_clean = df.dropna()
        df_sorted = df_clean.sort_index(axis=1)

        # Correlation matrix
        corr = df_sorted.corr()

        # Plot
        sns.set(style="white")

        plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        mask = np.triu(
            np.ones_like(corr, dtype=np.bool)
        )  # generate a mask for the upper triangle
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )

        plt.title(names[i] + "\n")

    plt.show()


def eof_correlation(eof_filepath, mask_filepath):
    """ Returns plot and DataArray of areas with p<0.05 """

    print("processing precipitation")
    da = dd.download_data(mask_filepath, xarray=True)
    tp_ds = da.mean(dim=["latitude", "longitude"]).tp
    tp = tp_ds.assign_coords(time=(tp_ds.time.astype("datetime64")))
    tp_df = tp.to_dataframe()

    print("processing EOF")
    eof_da = xr.open_dataset(eof_filepath)
    eof_ds = eof_da.EOF
    eof = eof_ds.assign_coords(time=(eof_ds.time.astype("datetime64")))
    eof_df = eof.to_dataframe()
    eof_pv = pd.pivot_table(
        eof_df, values="EOF", index=["time"], columns=["latitude", "longitude"]
    )
    eof_reset = eof_pv.reset_index()
    eof_reset["time"] -= np.timedelta64(12, "h")

    print("combining")
    df_combined = pd.merge_ordered(tp_df, eof_reset, on="time")
    df_clean = df_combined.dropna()

    corr_s = df_clean.corrwith(df_clean["tp"])
    corr_df = corr_s.to_frame(name="corr")
    corr_df["pvalue"] = pvalue(df_clean)

    filepath = "Data/EOF_corr_pval.csv"
    corr_df.to_csv(filepath)

    return filepath


def eof_correlation_map(corr_filepath):

    raw_df = pd.read_csv(corr_filepath, names=["coords", "corr", "pvalue"])
    corr_df = raw_df[2:]

    corr_array = corr_df["corr"].values.reshape(721, 1440).astype(float)
    pval_array = corr_df["pvalue"].values.reshape(721, 1440).astype(float)
    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(-180, 180, 1440)

    corr_da = xr.DataArray(
        data=corr_array, coords=[("latitude", lat), ("longitude", lon)]
    )
    pval_da = xr.DataArray(
        data=pval_array, coords=[("latitude", lat), ("longitude", lon)]
    )

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree())
    corr_da.plot(
        cbar_kwargs={"label": "\n Correlation", "extend": "neither", "pad": 0.10}
    )
    pval_da.plot.contour(levels=[0.05])
    ax.add_feature(cf.BORDERS)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()


def pvalue(df):
    """ Returns array of pvalues """
    df1 = df.drop(["time"], axis=1)
    pval_list = []

    for c in tqdm(list(df1)):
        _, pval = sp.stats.pearsonr(df1[c], df1["tp"])
        pval_list.append(pval)

    return np.array(pval_list)


def dataset_correlation(timeseries):
    """ Plots correlation heatmap for datasets """

    dataframes = []
    
    for ts in timeseries:
        df1 = ts.tp.to_dataframe(name=ts.plot_legend)
        df2 = df1.reset_index()
        df3 = df2.drop(["time", "lon", "lat"], axis=1)
        dataframes.append(df3)
    
    combined_df = dataframes[0].join(dataframes[1:]) 
    corr = combined_df.corr()

    sns.set(style="white")
    plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    mask = np.triu(
        np.ones_like(corr, dtype=np.bool)
    )  # generate a mask for the upper triangle
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        fmt="0.2f",
        square=True,
        linewidths=0.5,
        annot=True,
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.5},
    )
    plt.title("Correlation plot for gridded datasets")

    plt.show()