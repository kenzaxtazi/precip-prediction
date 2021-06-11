"""
Raw and bias corrected (Bannister et al.) WRF output.
"""

import xarray as xr
import numpy as np
from scipy.interpolate import griddata

import LocationSel as ls

def wrf_download(lon, lat):

    ds = xr.open_dataset('Data/Bannister_WRF_raw.nc')
    wrf_ds = ds.interp(coords={"x": lon, "y": lat}, method="nearest")
    wrf_ds = wrf_ds.assign_attrs(plot_legend="WRF")
    return wrf_ds


def bc_wrf_download(location):
    ds = xr.open_dataset('Data/Bannister_WRF_corrected.nc')
    bc_wrf_ds = ls.select_basin(ds, location)
    bc_wrf_ds = bc_wrf_ds.assign_attrs(plot_legend="Bias-corrected WRF")
    return bc_wrf_ds
    

def reformat_bannister_data():
    """ Saves data from Bannister et al with lon and lat instead of projections """

    wrf_ds = xr.open_dataset('Data/Bannister_WRF.nc')
    XLAT =  wrf_ds.XLAT.values
    XLONG = wrf_ds.XLONG.values
    m_precip = wrf_ds.model_precipitation.values
    bias_corr_precip = wrf_ds.bias_corrected_precipitation.values
    time = wrf_ds.time.values

    ds = xr.Dataset(data_vars=dict(m_precip=(["time", "x", "y"], m_precip), bias_corr_precip=(["time", "x", "y"], bias_corr_precip)),
                coords=dict(lon=(["x", "y"], XLONG), lat=(["x", "y"], XLAT), time=time))
    ds2 = (ds.resample(time="M")).mean()
    ds2['time'] = ds2.time.astype(float)/365/24/60/60/1e9 +1970

    # Raw WRF data
    wrf_ds = ds2.drop('bias_corr_precip')
    wrf_ds = wrf_ds.rename({'m_precip':'tp'})
    wrf_ds = interp(wrf_ds)
    wrf_ds.to_netcdf('Data/Bannister_WRF_raw.nc')

    # Bias corrected WRF data
    bc_ds = ds2.drop('m_precip')
    bc_ds = bc_ds.rename({'bias_corr_precip':'tp'})
    bc_ds = interp(bc_ds)
    bc_ds.to_netcdf('Data/Bannister_WRF_corrected.nc')


def interp(da):
    """ 
    Interpolation scheme to match data to ERA5 grid.
    Input:
        da (xarray DataArray): data we want to regrid
        mask_ds
    """

    # Generate a regular grid to interpolate the data
    x = np.arange(70, 85, 0.25)
    y = np.arange(25, 35, 0.25)
    grid_x, grid_y = np.meshgrid(y, x)

    # Create point coordinate pairs 
    lats = da.lat.values.flatten()
    lons = da.lon.values.flatten()
    times = da.time.values
    tp = da.tp.values.reshape(396, -1) 
    points = np.stack((lats, lons), axis=-1)

    # Interpolate using nearest neigbours
    grid_list = []
    for values in tp:
        grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_list.append(grid_z0)
    interp_grid = np.array(grid_list)

    # Turn into xarray DataSet
    ds = xr.Dataset(data_vars=dict(tp=(["time", "lon", "lat"], interp_grid)),
                    coords=dict(lon=(["lon"], x), lat=(["lat"], y), time= (['time'],times)))
    return ds





