# Collection of map plotting functions

import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.cm as cm
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from shapely.geometry import Polygon, shape, LinearRing
from cartopy.io import shapereader
from cartopy import config
from scipy import signal

import FileDownloader as fd
import DataPreparation as dp
import DataDownloader as dd

data_filepath = fd.update_cds_monthly_data()
mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"

"""
tp_filepath = 'Data/era5_tp_monthly_1979-2019.nc'
mpl_filepath = 'Data/era5_msl_monthly_1979-2019.nc'
"""


def annual_map(data_filepath, mask_filepath, variable, year, cumulative=False):
    """ Annual map """

    da = dd.apply_mask(data_filepath, mask_filepath)
    ds_year = da.sel(time=slice(str(year) + "-01-16T12:00:00", str(year + 1) + "-01-01T12:00:00"))
    ds_var = ds_year[variable] * 1000 # to mm/day 

    if cumulative is True:
        ds_processed = cumulative_monthly(ds_var)
        ds_final = ds_processed.sum(dim="time")
    else:
        ds_final = ds_var.std(dim="time")  # TODO weighted mean
    
    print(ds_final)

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    g = ds_final["tp_0001"].plot(cmap="magma_r", vmin=0.001, cbar_kwargs={"label": "Precipitation standard deviation [mm/day]", "extend": "neither", "pad": 0.10})
    g.cmap.set_under("white")
    #ax.add_feature(cf.BORDERS)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title("Upper Indus Basin Total Precipitation " + str(year) + "\n \n")
    plt.show()


def change_maps(data_filepath, mask_filepath, variable):
    """ Maps of average annual change from 1979 to 1989, 1999, 2009 and 2019 """

    da = dd.apply_mask(data_filepath, mask_filepath)
    da_var = da[variable] * 1000 # to mm/day 

    da_1979 = da_var.sel(
        time=slice("1979-01-16T12:00:00", str( + 1) + "1980-01-01T12:00:00")
    )
    da_processed = cumulative_monthly(da_1979)
    basin_1979_sum = da_processed.sum(dim="time")

    basin_1989 = da_var.sel(time=slice("1989-01-01T12:00:00", "1990-01-01T12:00:00"))
    basin_1989_sum = cumulative_monthly(basin_1989).sum(dim="time")
    basin_1989_change = basin_1989_sum / basin_1979_sum - 1

    basin_1999 = da_var.sel(time=slice("1999-01-01T12:00:00", "2000-01-01T12:00:00"))
    basin_1999_sum = cumulative_monthly(basin_1999).sum(dim="time")
    basin_1999_change = basin_1999_sum / basin_1979_sum - 1

    basin_2009 = da_var.sel(time=slice("2009-01-01T12:00:00", "2010-01-01T12:00:00"))
    basin_2009_sum = cumulative_monthly(basin_2009).sum(dim="time")
    basin_2009_change = basin_2009_sum / basin_1979_sum - 1

    basin_2019 = da_var.sel(time=slice("2019-01-01T12:00:00", "2020-01-01T12:00:00"))
    basin_2019_sum = cumulative_monthly(basin_2019).sum(dim="time")
    basin_2019_change = basin_2019_sum / basin_1979_sum - 1

    basin_changes = xr.concat(
        [basin_1989_change, basin_1999_change, basin_2009_change, basin_2019_change],
        pd.Index(["1989", "1999", "2009", "2019"], name="year"),
    )
    
    g = basin_changes.plot(
        x="longitude",
        y="latitude",
        col="year",
        col_wrap=2,
        subplot_kws={"projection": ccrs.PlateCarree()},
        cbar_kwargs={
            "label": "Precipitation change",
            "format": tck.PercentFormatter(xmax=1.0),
        },
    )

    for ax in g.axes.flat:
        ax.coastlines()
        ax.gridlines()
        ax.set_extent([71, 83, 30, 38])
        ax.add_feature(cf.BORDERS)

    plt.show()


def global_map(shapefiles=["Data/Shapefiles/UpperIndus_HP_shapefile/UpperIndus_HP.shp"]):
    """ 
    Returns global map with study area(s) superimposed.
    """
    # Shapefiles

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree(central_longitude=70))
    ax.coastlines("50m", linewidth=0.5)

    for s in shapefiles:
        path = s
        basin_shape = shapereader.Reader(path)

        for rec in basin_shape.records():
            ax.add_geometries(
                [rec.geometry],
                ccrs.AlbersEqualArea(
                    central_longitude=125, central_latitude=-15, standard_parallels=(7, -32)
                ),
                edgecolor="red",
                facecolor="red",
                alpha=0.1,
            )

    plt.show()


def indus_map(uib_only=False, bs_only=False):
    """ Returns two maps: global map with box, zoomed in map of the Upper Indus Basin """

    # River shapefiles
    fpath = "Data/Shapefiles/ne_50m_rivers_lake_centerlines_scale_rank/ne_50m_rivers_lake_centerlines_scale_rank.shp"
    as_shp = shapereader.Reader(fpath)

    # UIB shapefile
    uib_path = "Data/Shapefiles/UpperIndus_HP_shapefile/UpperIndus_HP.shp"
    uib_shape = shapereader.Reader(uib_path)

    # Beas + Sutlej shapefile and projection
    bs_path = "Data/Shapefiles/beas-sutlej-shapefile/12500Ha.shp"
    bs_shape = shapereader.Reader(bs_path)
    bs_globe = ccrs.Globe(semimajor_axis=6377276.345, inverse_flattening=300.8017)
    cranfield_crs = ccrs.LambertConformal(
                central_longitude=82, central_latitude=20, false_easting=2000000.0, false_northing=2000000.0,
                standard_parallels=[12.47294444444444,35.17280555555556], globe=bs_globe)

    # Other phyical features
    land_50m = cf.NaturalEarthFeature("physical", "land", "50m", edgecolor="face", facecolor=cf.COLORS["land"])
    ocean_50m = cf.NaturalEarthFeature("physical", "ocean", "50m", edgecolor="face", facecolor=cf.COLORS["water"])

    # Regional rectangle
    pgon = regional_rectangle(60, 80, 20, 40)

    # Projection for global map
    glb_proj = ccrs.NearsidePerspective(central_longitude=70, central_latitude=30)

    plt.figure("Global")
    ax = plt.subplot(projection=glb_proj)
    ax.add_feature(ocean_50m)
    ax.add_feature(land_50m)
    ax.coastlines("50m", linewidth=0.4)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
    ax.add_geometries([pgon], crs=ccrs.PlateCarree(), facecolor="none", edgecolor="red")

    plt.figure("Zoomed in")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([60, 85, 20, 40])

    ax.add_feature(land_50m)
    ax.add_feature(ocean_50m)
    ax.add_feature(cf.BORDERS.with_scale("50m"), linewidth=0.5)
    ax.coastlines("50m", linewidth=0.5)

    if bs_only == False:
        for rec in uib_shape.records():
            ax.add_geometries(
                [rec.geometry],
                ccrs.AlbersEqualArea(
                    central_longitude=125, central_latitude=-15, standard_parallels=(7, -32)
                ),
                edgecolor="red",
                facecolor="red",
                alpha=0.1,
            )

    if uib_only == False:
        for rec in bs_shape.records():
            ax.add_geometries(
                [rec.geometry],
                cranfield_crs,
                edgecolor=None,
                facecolor="green",
                alpha=0.1,
                )

    # ax.add_feature(rivers)
    for rec in as_shp.records():
        if rec.attributes["name"] == "Indus":
            ax.add_geometries(
                [rec.geometry],
                ccrs.PlateCarree(),
                edgecolor="steelblue",
                facecolor="None",
            )
        pass

    ax.text(64, 28, "PAKISTAN", c="gray")
    ax.text(62.5, 33.5, "AFGHANISTAN", c="gray")
    ax.text(80, 37, "CHINA", c="gray")
    ax.text(77, 25, "INDIA", c="gray")
    ax.text(62, 22, "Arabian Sea", c="navy", alpha=0.5)
    ax.text(70.3, 32, "Indus river", c="steelblue", rotation=55, size=7)

    ax.set_xticks([60, 65, 70, 75, 80, 85])
    ax.set_xticklabels(["60°E", "65°E", "70°E", "75°E", "80°E", "85°E"])

    ax.set_yticks([20, 25, 30, 35, 40])
    ax.set_yticklabels(["20°N", "25°N", "30°N", "35°N", "40°N"])

    plt.title("Indus River \n \n")
    plt.show()


def regional_rectangle(lonmin, lonmax, latmin, latmax, nvert=100):
    """ 
    Returns Polygon object to create regional rectangle on maps.
    """
    lons = np.r_[
        np.linspace(lonmin, lonmin, nvert),
        np.linspace(lonmin, lonmax, nvert),
        np.linspace(lonmax, lonmax, nvert),
    ].tolist()
    
    lats = np.r_[
        np.linspace(latmin, latmax, nvert),
        np.linspace(latmax, latmax, nvert),
        np.linspace(latmax, latmin, nvert),
    ].tolist()

    pgon = LinearRing(list(zip(lons, lats)))

    return pgon




def cumulative_monthly(da):
    """ Multiplies monthly averages by the number of day in each month """
    x, y, z = np.shape(da.values)
    times = np.datetime_as_string(da.time.values)
    days_in_month = []
    for t in times:
        year = t[0:4]
        month = t[5:7]
        days = calendar.monthrange(int(year), int(month))[1]
        days_in_month.append(days)
    dim = np.array(days_in_month)
    dim_mesh = np.repeat(dim, y * z).reshape(x, y, z)

    return da * dim_mesh

