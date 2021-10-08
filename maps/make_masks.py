# Adapted from Tony Phillips code


import os
import iris
import numpy as np
import pandas as pd
import xarray as xr

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from shapely.geometry import Polygon
from shapely.ops import unary_union


# read a template ERA5 field
mask = xr.open_dataset('_Data/SRTM_data.nc')
mask = mask.rename({'nlat': 'lat', 'nlon': 'lon'})

# calculate the coordinates for the grid cell boundaries
xb = mask.lon.values
yb = mask.lat.values

# create a mesh of the grid cell boundaries
xb2, yb2 = np.meshgrid(xb, yb)

# create a CRS for the Cranfield model
# parameters from file "All_EBands_NoEndorreic.prj"
cranfield_cs = iris.coord_systems.GeogCS(
    semi_major_axis=6377276.345, inverse_flattening=300.8017)
cranfield_proj = iris.coord_systems.LambertConformal(
    central_lat=20.0, central_lon=82.0,
    false_easting=2000000.0, false_northing=2000000.0,
    secant_latitudes=[12.47294444444444, 35.17280555555556],
    ellipsoid=cranfield_cs)
cranfield_crs = cranfield_proj.as_cartopy_crs()

# project the lons and lats into the Cranfield CRS
cr_xyz = cranfield_crs.transform_points(
    src_crs=ccrs.PlateCarree(), x=xb2, y=yb2)
crx = cr_xyz[:, :, 0]
cry = cr_xyz[:, :, 1]

# create a template overlap field
template = mask[['lon', 'lat']]
template = template.assign(Overlap=mask.slope * 0)

# get the grid size in X and Y
nx = template.Overlap.values.shape[1]
ny = template.Overlap.values.shape[0]

# calculate overlaps for SWAT shapefile
swat_reader = shpreader.Reader(
    '_Data/Shapefiles/beas-sutlej-shapefile/12500Ha.shp')

# read the subbasin-to-basin map
map = pd.read_csv(
    '_Data/Shapefiles/beas-sutlej-shapefile/SWAT_subbasin_basins.csv')

# for each basin...
polys = []
all_records = swat_reader.records()
for record in all_records:
    subbasin = record.attributes['Subbasin']
    polys.append(record.geometry)

poly = unary_union(polys)

overlap = template.copy()
envelope = poly.envelope
for x in range(0, nx-1):
    for y in range(0, ny-1):
        polygon = Polygon([(crx[y, x], cry[y, x]), (crx[y+1, x], cry[y+1, x]),
                           (crx[y+1, x+1], cry[y+1, x+1]),
                           (crx[y, x+1], cry[y, x+1])])
        if polygon.is_valid and envelope.intersects(polygon):
            overlap['Overlap'][y, x] = polygon.intersection(
                poly).area / polygon.area

# if any values are fractionally above 1, make them 1
overlap['Overlap'].where((abs(overlap['Overlap'] - 1) > 1e-05 +
                         1e-08 * overlap['Overlap']) |
                         (overlap['Overlap'] < 1), 1)

# zero all values in the SH
overlap['Overlap'][template.coord('latitude').points < 0, :] = 0
mask_filepath = '_Data/Masks/Beas_Sutlej_highres_mask.nc'
overlap.to_netcdf(mask_filepath)
