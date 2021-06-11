"""
Raw and bias corrected (Bannister et al.) WRF output.
"""

import xarray as xr


def wrf_download(lon, lat):

    ds = xr.open_dataset('Data/WRF_corrected.nc')
    ds = ds.drop('bias_corr_precip')
    wrf_ds = ds.rename({'m_precip':'tp'})
    wrf_ds = wrf_ds.interp(coords={"x": lon, "y": lat}, method="nearest")
    wrf_ds = wrf_ds.assign_attrs(plot_legend="WRF")
    
    return wrf_ds


def bc_wrf_download(lon, lat):

    ds = xr.open_dataset('Data/WRF_corrected.nc')
    ds = ds.drop('m_precip')
    bc_wrf_ds = ds.rename({'bias_corr_precip':'tp'})
    bc_wrf_ds = bc_wrf_ds.interp(coords={"x": lon, "y": lat}, method="nearest")
    bc_wrf_ds = bc_wrf_ds.assign_attrs(plot_legend=" Bias-corrected WRF")
    
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
    ds2 = ds.resample(time="M").mean()
    print(ds2)
    ds2['time'] = ds2.time.astype(float)/365/24/60/60/1e9 +1970

    ds2.to_netcdf('Data/WRF_corrected.nc')


