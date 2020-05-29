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

import PrecipitationDataExploration as pde
import FileDownloader as fd
import Clustering as cl

# Filepaths and URLs

mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'

nao_url = 'https://www.psl.noaa.gov/data/correlation/nao.data'
n34_url =  'https://psl.noaa.gov/data/correlation/nina34.data'
n4_url =  'https://psl.noaa.gov/data/correlation/nina4.data'


# Front indices

nao_df = fd.update_url_data(nao_url, 'NAO')
n34_df = fd.update_url_data(n34_url, 'N34')
n4_df = fd.update_url_data(n4_url, 'N4')

ind_df = nao_df.join([n34_df, n4_df]).astype('float64')


# Orography, humidity and precipitation

cds_filepath = fd.update_cds_data(variables=['2m_dewpoint_temperature', 'angle_of_sub_gridscale_orography', 
                                              'orography', 'slope_of_sub_gridscale_orography', 
                                              'total_column_water_vapour', 'total_precipitation'])

masked_da = pde.apply_mask(cds_filepath, mask_filepath)
multiindex_df = masked_da.to_dataframe()
cds_df = multiindex_df.reset_index()

## Create clustered dataframes


## Concatonate dataframes together
df_combined = pd.merge_ordered(cds_df, ind_df, on='time')  
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

plt.title('Correlation plot')
plt.show()
