# Trend comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import cftime
import datetime

import DataDownloader as dd



# ERA5 
mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"
era5_ds= dd.download_data(mask_filepath, xarray=True)

era5_utime = era5_ds.time.values / (1e9 * 60 * 60 * 24 * 365)


# CMIP5 historical
cmip_59_84_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_Amon_HadCM3_historical_r1i1p1_195912-198411.nc")
cmip_84_05_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_Amon_HadCM3_historical_r1i1p1_198412-200512.nc")
cmip_ds = cmip_84_05_ds.merge(cmip_59_84_ds)

cmip_time = np.array([d.strftime() for d in cmip_ds.time.values])
cmip_time2 = np.array([datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in cmip_time])
cmip_utime = np.array([d.timestamp() for d in cmip_time2])/ ( 60 * 60 * 24 * 365)


# CORDEX 
cordex_90_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_199001-199012.nc")
cordex_91_00_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_199101-200012.nc")
cordex_01_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_200101-201012.nc")
cordex_02_11_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_201101-201111.nc")
cordex_90_00_ds = cordex_90_ds.merge(cordex_91_00_ds)
cordex_01_11_ds= cordex_01_ds.merge(cordex_02_11_ds)
cordex_ds = cordex_01_11_ds.merge(cordex_90_00_ds)

cordex_time = cordex_ds.time.values.astype(int)
cordex_utime = cordex_time / (1e9 * 60 * 60 * 24 * 365)


# Gilgit station 34°N 78°E
era5_gilgit_ds = era5_ds.interp(coords={"longitude": 78, "latitude": 34}, method="nearest")
cmip_gilgit_ds = cmip_ds.interp(coords={"lon": 78, "lat": 34}, method="nearest")
cordex_gilgit_ds = cordex_ds.interp(coords={"lon": 78, "lat": 34}, method="nearest")



# Plot
plt.figure()

plt.plot(cmip_utime + 1970, cmip_gilgit_ds.pr.values * 60 * 60 * 24, c='b', label='CMIP5 HAD3CM')
plt.plot(cordex_utime + 1970, cordex_gilgit_ds.pr.values * 60 * 60 * 24, c='g', label='CORDEX East Asia')
plt.plot(era5_utime + 1970, era5_gilgit_ds.tp.values * 1000, c='r', label='ERA5')
plt.xlabel('Time')
plt.ylabel('Precipitation mm/day')
plt.legend()
plt.show()














