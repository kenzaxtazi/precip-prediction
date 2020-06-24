import matplotlib
matplotlib.use('agg')

import datetime
import urllib

import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import DataExploration as de
import DataPreparation as dp
import FileDownloader as fd
import Clustering as cl


### Filepaths
mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'
data_filepath = fd.update_cds_data()


### Upper Indus Basin correlation plot 

df = dp.download_data(mask_filepath)

# create lags
df['NAO-1'] = df['NAO'].shift(periods=1)
df['NAO-2'] = df['NAO'].shift(periods=2)
df['NAO-3'] = df['NAO'].shift(periods=3)
df['NAO-4'] = df['NAO'].shift(periods=4)
df['NAO-5'] = df['NAO'].shift(periods=5)
df['NAO-6'] = df['NAO'].shift(periods=6)

df['N34-1'] = df['N34'].shift(periods=1)
df['N34-2'] = df['N34'].shift(periods=2)
df['N34-3'] = df['N34'].shift(periods=3)
df['N34-4'] = df['N34'].shift(periods=4)
df['N34-5'] = df['N34'].shift(periods=5)
df['N34-6'] = df['N34'].shift(periods=6)

df['N4-1'] = df['N4'].shift(periods=1)
df['N4-2'] = df['N4'].shift(periods=2)
df['N4-3'] = df['N4'].shift(periods=3)
df['N4-4'] = df['N4'].shift(periods=4)
df['N4-5'] = df['N4'].shift(periods=5)
df['N4-6'] = df['N4'].shift(periods=6)

df = df.drop(columns=['expver', 'Unnamed: 0'])
df_clean = df.dropna()
df_sorted = df_clean.sort_index(axis=1)


# Correlation matrix
corr = df_sorted.corr()  

# Plot
sns.set(style="white")

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
mask = np.triu(np.ones_like(corr, dtype=np.bool))  # generate a mask for the upper triangle
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlation plot for Upper Indus Basin')
plt.show()



### Cluster correlation plot 

n_clusters = 3
masked_da = dp.apply_mask(data_filepath, mask_filepath) # TODO fix to work with download_data()

nao_url = 'https://www.psl.noaa.gov/data/correlation/nao.data'
n34_url =  'https://psl.noaa.gov/data/correlation/nina34.data'
n4_url =  'https://psl.noaa.gov/data/correlation/nina4.data'

# Indices
nao_df = fd.update_url_data(nao_url, 'NAO')
n34_df = fd.update_url_data(n34_url, 'N34')
n4_df = fd.update_url_data(n4_url, 'N4')
ind_df = nao_df.join([n34_df, n4_df]).astype('float64')

cds_da_clusters = cl.gp_clusters(masked_da.tp, N=n_clusters, filter=0.7, plot=True)
names =['Gilgit regime', 'Ngari regime', 'Khyber regime']

for i in range(n_clusters):
    cluster_da = masked_da.where(cds_da_clusters[i] >= 0)
    multiindex_df = cluster_da.to_dataframe()
    cluster_df = multiindex_df.reset_index()
   
    df_combined = pd.merge_ordered(cluster_df, ind_df, on='time')

    # create lags
    df_combined['NAO-1'] = df_combined['NAO'].shift(periods=1)
    df_combined['NAO-2'] = df_combined['NAO'].shift(periods=2)
    df_combined['NAO-3'] = df_combined['NAO'].shift(periods=3)
    df_combined['NAO-4'] = df_combined['NAO'].shift(periods=4)
    df_combined['NAO-5'] = df_combined['NAO'].shift(periods=5)
    df_combined['NAO-6'] = df_combined['NAO'].shift(periods=6)

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

    df = df_combined.drop(columns=['expver'])
    df_clean = df.dropna()
    df_sorted = df_clean.sort_index(axis=1)

    # Correlation matrix
    corr = df_sorted.corr()  

    # Plot
    sns.set(style="white")

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    mask = np.triu(np.ones_like(corr, dtype=np.bool))  # generate a mask for the upper triangle
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title(names[i] + '\n')

plt.show()

