"""
SRTM dataset is a 90m DEM computed from satellite.

Slope and aspect are calculated using:
    Horn, B.K.P., 1981. Hill shading and the reflectance map. Proceedings of
    the IEEE 69, 14–47. doi:10.1109/PROC.1981.11918
"""

import numpy as np
import pandas as pd
import xarray as xr
import richdem as rd


def generate_slope_aspect():
    """Generate slope and aspect from DEM."""
    dem_ds = xr.open_dataset(
        '/Users/kenzatazi/Downloads/GMTED2010_15n015_00625deg.nc')
    dem_ds = dem_ds.assign_coords(
        {'nlat': dem_ds.latitude, 'nlon': dem_ds.longitude})
    dem_ds = dem_ds.sel(nlat=slice(29, 34), nlon=slice(75, 83))

    elev_arr = dem_ds.elevation.values
    elev_rd_arr = rd.rdarray(elev_arr, no_data=np.nan)

    slope_rd_arr = rd.TerrainAttribute(elev_rd_arr, attrib='slope_riserun')
    slope_arr = np.array(slope_rd_arr)

    aspect_rd_arr = rd.TerrainAttribute(elev_rd_arr, attrib='aspect')
    aspect_arr = np.array(aspect_rd_arr)

    dem_ds['slope'] = (('nlat', 'nlon'), slope_arr)
    dem_ds['aspect'] = (('nlat', 'nlon'), aspect_arr)

    streamlined_dem_ds = dem_ds[['elevation', 'slope', 'aspect']]
    streamlined_dem_ds.to_netcdf('_Data/SRTM_data.nc')


def find_slope(station):
    """Return slope for given station."""
    dem_ds = xr.open_dataset('Data/SRTM_data.nc')
    all_station_dict = pd.DataFrame('Data/gauge_info.csv')
    location = all_station_dict[station]
    station_slope = dem_ds.interp(
        coords={"nlon": location[1], "nlat": location[0]}, method="nearest")
    return station_slope
