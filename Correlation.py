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
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'
data_filepath = fd.update_cds_data()


### Upper Indus Basin correlation plot 

df = dp.download_data(mask_filepath)

# create lags
df['NAO+1'] = df['NAO'].shift(periods=-1)
df['NAO+2'] = df['NAO'].shift(periods=-2)
df['NAO+3'] = df['NAO'].shift(periods=-3)
df['NAO+4'] = df['NAO'].shift(periods=-4)
df['NAO+5'] = df['NAO'].shift(periods=-5)
df['NAO+6'] = df['NAO'].shift(periods=-6)

df['N34+1'] = df['N34'].shift(periods=-1)
df['N34+2'] = df['N34'].shift(periods=-2)
df['N34+3'] = df['N34'].shift(periods=-3)
df['N34+4'] = df['N34'].shift(periods=-4)
df['N34+5'] = df['N34'].shift(periods=-5)
df['N34+6'] = df['N34'].shift(periods=-6)

df['N4+1'] = df['N4'].shift(periods=-1)
df['N4+2'] = df['N4'].shift(periods=-2)
df['N4+3'] = df['N4'].shift(periods=-3)
df['N4+4'] = df['N4'].shift(periods=-4)
df['N4+5'] = df['N4'].shift(periods=-5)
df['N4+6'] = df['N4'].shift(periods=-6)

df = df.drop(columns=['expver', 'Unnamed: 0'])
df_clean = df.dropna()

# Correlation matrix
corr = df_clean.corr()  

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


'''
### Cluster correlation plot 

n_clusters = 3
masked_da = dp.apply_mask(data_filepath, mask_filepath) # TODO fix to work with download_data()

cds_da_clusters = cl.gp_clusters(masked_da.tp, N=n_clusters, filter=0.7, plot=True)

for i in range(n_clusters):
    cluster_da = masked_da.where(cds_da_clusters[i] >= 0)
    multiindex_df = cluster_da.to_dataframe()
    cluster_df = multiindex_df.reset_index()
   
    df_combined = pd.merge_ordered(cluster_df, ind_df, on='time')  
    df = df_combined.drop(columns=['expver'])
    df_clean = df.dropna()

    # Correlation matrix
    corr = df_clean.corr()  

    # Plot
    sns.set(style="white")

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    mask = np.triu(np.ones_like(corr, dtype=np.bool))  # generate a mask for the upper triangle
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Cluster '+ str(i) + '\n')
'''

plt.show()


def detrend():  #TODO
    return 1

