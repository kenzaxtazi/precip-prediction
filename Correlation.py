
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import urllib
import PrecipitationDataExploration as pde
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
import cartopy.feature as cf
import pandas as pd
import seaborn as sns
import datetime


# Clusters


# Front indices

filepath = '/Users/kenzatazi/Downloads/'
now = datetime.datetime.now()

nao_url = 'https://www.psl.noaa.gov/data/correlation/nao.data'
nao_file = filepath + 'nao-' + now.strftime("%m-%Y") + '.csv'

n34_url =  'https://psl.noaa.gov/data/correlation/nina34.data'
n34_file = filepath + 'n34-' + now.strftime("%m-%Y") + '.csv'

n4_url =  'https://psl.noaa.gov/data/correlation/nina4.data'
n4_file = filepath + 'n4-' + now.strftime("%m-%Y") + '.csv'

## open as dataframe 
nao_df = pde.update_data(nao_url, nao_file, 'NAO')
n34_df = pde.update_data(n34_url, n34_file, 'N34')
n4_df = pde.update_data(n4_url, n4_file, 'N4')


# Orography and humidity



# Concatonate pds together
df = precipitation_df.join([nao_df, n34_df, n4_df])
df=df.astype('float64')

# Correlation matrix
corr = df.corr()  


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
