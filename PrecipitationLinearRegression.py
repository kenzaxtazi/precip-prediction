# PrecipitationLinearRegression

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import xarray as xr

import FileDownloader as fd
import DataDownloader as dd

from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Open data
mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'
tp= dd.download_data(mask_filepath, xarray=True)
tp_da = tp.tp *1000  # convert from m/day to mm/day 

# Sklearn model for location timeseries

## Data 
gilgit = tp_da.interp(coords={'longitude':75, 'latitude':36}, method='nearest') 
ngari = tp_da.interp(coords={'longitude':81, 'latitude':32}, method='nearest') 
khyber = tp_da.interp(coords={'longitude':73, 'latitude':34.5}, method='nearest')

gilgit_values = gilgit.values
ngari_values = ngari.values
khyber_values = khyber.values

stime = gilgit.time.values
utime = (stime.astype('int'))/(1e9*60*60*24*365)  # from numpy.datetime64 to unix timestamp


## Models
gilgit_slope, gilgit_intercept, gilgit_r_value, gilgit_p_value, gilgit_std_err = stats.linregress(utime, gilgit_values)
ngari_slope, ngari_intercept, ngari_r_value, ngari_p_value, ngari_std_err = stats.linregress(utime, ngari_values)
khyber_slope, khyber_intercept, khyber_r_value, khyber_p_value, khyber_std_err = stats.linregress(utime, khyber_values)

'''

## Predictions
gilgit_pred = gilgit_model.predict(utime)
ngari_pred = ngari_model.predict(utime)
khyber_pred = khyber_model.predict(utime)

## Coefficients 
print('Gilgit coefficients: \n', gilgit_model.coef_*1e9*60*60*24*365) # to mm/day/year 
print('ngari coefficients: \n', ngari_model.coef_*1e9*60*60*24*365)
print('khyber coefficients: \n', khyber_model.coef_*1e9*60*60*24*365)

print('Gilgit coefficient of determination: %.5f' % r2_score(gilgit_values, gilgit_pred))
print('ngari coefficient of determination: %.5f' % r2_score(ngari_values, ngari_pred))
print('khyber coefficient of determination: %.5f' % r2_score(khyber_values, khyber_pred))
'''

print(gilgit_p_value, ngari_p_value, khyber_p_value)

## Plots 
fig, axs = plt.subplots(3, sharex=True, sharey=True)

axs[0].plot(utime+1970, gilgit_values)
axs[0].set_title('36.00°N 75.00°E')
axs[0].plot(utime+1970, gilgit_slope * utime + gilgit_intercept, color='green', linestyle='--', 
            label='Slope = %.3f±%.3f mm/day/year, p-value = %.3f' % (gilgit_slope, gilgit_std_err, gilgit_p_value))
axs[0].set_xlabel(' ')
axs[0].set_ylabel('Total precipation [mm/day]')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(utime+1970, ngari_values)
axs[1].set_title('32.00°N 81.00°E')
axs[1].plot(utime+1970, ngari_slope * utime + ngari_intercept, color='green', linestyle='--',
            label='Slope = %.3f±%.3f mm/day/year, p-value = %.3f' % (ngari_slope, ngari_std_err, ngari_p_value))
axs[1].set_xlabel(' ')
axs[1].set_ylabel('Total precipation [mm/day]')
axs[1].grid(True)
axs[1].legend()

axs[2].plot(utime+1970, khyber_values)
axs[2].set_title('34.50°N 73.00°E')
axs[2].plot(utime+1970, khyber_slope * utime + khyber_intercept, color='green', linestyle='--',
            label='Slope = %.3f±%.3f mm/day/year, p-value = %.3f' % (khyber_slope, khyber_std_err, khyber_p_value))
axs[2].set_xlabel('Year')
axs[2].set_ylabel('Total precipation [mm/day]')
axs[2].grid(True)
axs[2].legend()

plt.show()
