import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import xarray as xr

import FileDownloader as fd 

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Open data

tp_filepath = 'era5_tp_monthly_1979-2019.nc'
mpl_filepath = 'era5_msl_monthly_1979-2019.nc.download'

tp= xr.open_dataset(tp_filepath)
tp_da = tp.tp *1000  # convert from m/day to mm/day 

# Sklearn model for location timeseries

## Data 
gilgit = tp_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest') 
skardu = tp_da.interp(coords={'longitude':75.5550, 'latitude':35.3197 }, method='nearest') 
leh = tp_da.interp(coords={'longitude':77.5706, 'latitude':34.1536 }, method='nearest')

gilgit_values = gilgit.values
skardu_values = skardu.values
leh_values = leh.values

stime = gilgit.time.values
utime = (stime.astype('int')).reshape(-1, 1)  # from numpy.datetime64 to unix timestamp


## Models
gilgit_model = linear_model.LinearRegression()
skardu_model = linear_model.LinearRegression()
leh_model = linear_model.LinearRegression()

## Fitting
gilgit_model.fit(utime, gilgit_values)
skardu_model.fit(utime, skardu_values)
leh_model.fit(utime, leh_values)

## Predictions
gilgit_pred = gilgit_model.predict(utime)
skardu_pred = skardu_model.predict(utime)
leh_pred = leh_model.predict(utime)

## Coefficients 
print('Gilgit coefficients: \n', gilgit_model.coef_*1e9*60*60*24*365) # to mm/day/year 
print('Skardu coefficients: \n', skardu_model.coef_*1e9*60*60*24*365)
print('Leh coefficients: \n', leh_model.coef_*1e9*60*60*24*365)

print('Gilgit coefficient of determination: %.5f' % r2_score(gilgit_values, gilgit_pred))
print('Skardu coefficient of determination: %.5f' % r2_score(skardu_values, skardu_pred))
print('Leh coefficient of determination: %.5f' % r2_score(leh_values, leh_pred))

## Plots 
fig, axs = plt.subplots(3, sharex=True, sharey=True)

gilgit.plot(ax=axs[0])
axs[0].set_title('Gilgit (35.8884°N, 74.4584°E, 1500m)')
axs[0].plot(stime, gilgit_pred, color='green', linestyle='--', 
            label='Slope = %.3f mm/day/year' % (gilgit_model.coef_*1e9*60*60*24*365*1000))
axs[0].set_xlabel(' ')
axs[0].set_ylabel('Total precipation [mm/day]')
axs[0].grid(True)
axs[0].legend()

skardu.plot(ax=axs[1])
axs[1].set_title('Skardu (35.3197°N, 75.5550°E, 2228m)')
axs[1].plot(stime, skardu_pred, color='green', linestyle='--',
            label='Slope = %.3f mm/day/year' % (skardu_model.coef_*1e9*60*60*24*365*1000))
axs[1].set_xlabel(' ')
axs[1].set_ylabel('Total precipation [mm/day]')
axs[1].grid(True)
axs[1].legend()

leh.plot(ax=axs[2])
axs[2].set_title('Leh (34.1536°N, 77.5706°E, 3500m)')
axs[2].plot(stime, leh_pred, color='green', linestyle='--',
            label='Slope = %.3f mm/day/year' % (leh_model.coef_*1e9*60*60*24*365*1000))
axs[2].set_xlabel('Year')
axs[2].set_ylabel('Total precipation [mm/day]')
axs[2].grid(True)
axs[2].legend()

plt.show()
