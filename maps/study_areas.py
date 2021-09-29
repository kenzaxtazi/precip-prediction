# Collection of map plotting functions


import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs

import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from cartopy.io import shapereader
from shapely.geometry import LinearRing


def global_map(uib_only=False, bs_only=False):
    """ Return global map with study area(s) superimposed."""
    # UIB shapefile
    uib_path = "_Data/Shapefiles/UpperIndus_HP_shapefile/UpperIndus_HP.shp"
    uib_shape = shapereader.Reader(uib_path)

    # Beas + Sutlej shapefile and projection
    bs_path = "_Data/Shapefiles/beas-sutlej-shapefile/12500Ha.shp"
    bs_shape = shapereader.Reader(bs_path)
    bs_globe = ccrs.Globe(semimajor_axis=6377276.345,
                          inverse_flattening=300.8017)
    cranfield_crs = ccrs.LambertConformal(
        central_longitude=82, central_latitude=20, false_easting=2000000.0,
        false_northing=2000000.0, standard_parallels=[12.47294444444444,
                                                      35.17280555555556],
        globe=bs_globe)

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree(central_longitude=70))
    ax.coastlines("50m", linewidth=0.5)

    if bs_only is False:
        for rec in uib_shape.records():
            ax.add_geometries(
                [rec.geometry],
                ccrs.AlbersEqualArea(
                    central_longitude=125, central_latitude=-15,
                    standard_parallels=(7, -32)),
                edgecolor=None,
                facecolor="red",
                alpha=0.1)

    if uib_only is False:
        for rec in bs_shape.records():
            ax.add_geometries(
                [rec.geometry],
                cranfield_crs,
                edgecolor=None,
                facecolor="green",
                alpha=0.1)

    plt.show()


def indus_map():
    """Return two maps: global map with box, zoomed in map of the Upper Indus Basin."""

    # River shapefiles
    fpath = "_Data/Shapefiles/ne_50m_rivers_lake_centerlines_scale_rank/ne_50m_rivers_lake_centerlines_scale_rank.shp"
    as_shp = shapereader.Reader(fpath)

    # UIB shapefile
    uib_path = "_Data/Shapefiles/UpperIndus_HP_shapefile/UpperIndus_HP.shp"
    uib_shape = shapereader.Reader(uib_path)

    # Beas shapefile and projection
    beas_path = "_Data/Shapefiles/beas-sutlej-shapefile/beas_sutlej_basins/beas_watershed.shp"
    beas_shape = shapereader.Reader(beas_path)
    beas_globe = ccrs.Globe(semimajor_axis=6377276.345,
                            inverse_flattening=300.8017)
    beas_cranfield_crs = ccrs.LambertConformal(
        central_longitude=80, central_latitude=23, false_easting=0.0,
        false_northing=0.0, standard_parallels=[30, 30], globe=beas_globe)

    # Sutlej shapefile and projection
    stlj_path = "_Data/Shapefiles/beas-sutlej-shapefile/beas_sutlej_basins/sutlej_watershed.shp"
    stlj_shape = shapereader.Reader(stlj_path)
    stlj_globe = ccrs.Globe(semimajor_axis=6377276.345,
                            inverse_flattening=300.8017)
    stlj_cranfield_crs = ccrs.LambertConformal(
        central_longitude=80, central_latitude=23, false_easting=0.0,
        false_northing=0.0, standard_parallels=[30, 30], globe=stlj_globe)

    # Other phyical features
    land_50m = cf.NaturalEarthFeature(
        "physical", "land", "50m", edgecolor="face",
        facecolor=cf.COLORS["land"])
    ocean_50m = cf.NaturalEarthFeature(
        "physical", "ocean", "50m", edgecolor="face",
        facecolor=cf.COLORS["water"])

    # Regional rectangle
    pgon = regional_rectangle(60, 80, 20, 40)

    # Projection for global map
    glb_proj = ccrs.NearsidePerspective(
        central_longitude=70, central_latitude=30)

    plt.figure("Global")
    ax = plt.subplot(projection=glb_proj)
    ax.add_feature(ocean_50m)
    ax.add_feature(land_50m)
    ax.coastlines("50m", linewidth=0.4)
    ax.add_feature(cf.BORDERS, linewidth=0.4)
    ax.add_geometries([pgon], crs=ccrs.PlateCarree(),
                      facecolor="none", edgecolor="red")

    plt.figure("Zoomed in")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([60, 85, 20, 40])

    ax.add_feature(land_50m)
    ax.add_feature(ocean_50m)
    ax.add_feature(cf.BORDERS.with_scale("50m"), linewidth=0.5)
    ax.coastlines("50m", linewidth=0.5)

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
    for rec in beas_shape.records():
        ax.add_geometries(
            [rec.geometry],
            beas_cranfield_crs,
            edgecolor="green",
            facecolor="green",
            alpha=0.1,
        )

    for rec in stlj_shape.records():
        ax.add_geometries(
            [rec.geometry],
            stlj_cranfield_crs,
            edgecolor="blue",
            facecolor="blue",
            alpha=0.1,
        )

    # ax.add_feature(rivers)
    for rec in as_shp.records():
        if (rec.attributes["name"] == "Indus") | \
            (rec.attributes["name"] == 'Beas') | \
                (rec.attributes["name"] == 'Sutlej'):
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
    """ Return Polygon object to create regional rectangle on maps."""
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


def beas_sutlej_gauge_map():
    # TODO could include length of datasets as marker size
    """ Maps of gauges used by Bannister """

    station_df = pd.DataFrame.from_csv('Data/gauge_info')

    hf_train_df1 = station_df[(station_df['lon'] < 77.0)
                              & (station_df['lat'] > 32)]
    hf_train_df2 = station_df[(station_df['lon'] < 76.60) & (
        (station_df['lat'] < 32) & (station_df['lat'] > 31.6))]
    hf_train_df3 = station_df[(station_df['lon'] > 77.0)
                              & (station_df['lat'] < 31)]
    hf_train_df4 = station_df[(station_df['lon'] < 78.0) & (
        station_df['lon'] > 77.0) & (station_df['lat'] > 31)
        & (station_df['lat'] < 31.23)]
    hf_train_df5 = station_df[(station_df['lon'] > 78.2)]
    hf_train_stations = pd.concat(
        [hf_train_df1, hf_train_df2, hf_train_df3, hf_train_df4, hf_train_df5])

    # Gauges
    filepath = 'Data/stations_MGM.xlsx'
    df = pd.read_excel(filepath)

    # Beas and Sutlej shapefile and projection
    bs_path = "_Data/Shapefiles/beas-sutlej-shapefile/12500Ha.shp"
    bs_shape = shapereader.Reader(bs_path)
    bs_globe = ccrs.Globe(semimajor_axis=6377276.345,
                          inverse_flattening=300.8017)
    cranfield_crs = ccrs.LambertConformal(
        central_longitude=82, central_latitude=20, false_easting=2000000.0, false_northing=2000000.0,
        standard_parallels=[12.47294444444444, 35.17280555555556], globe=bs_globe)

    # Topography
    top_ds = xr.open_dataset(
        '/Users/kenzatazi/Downloads/GMTED2010_15n015_00625deg.nc')
    top_ds = top_ds.assign_coords(
        {'nlat': top_ds.latitude, 'nlon': top_ds.longitude})
    top_ds = top_ds.sel(nlat=slice(29, 34), nlon=slice(75, 83))

    # Figure
    plt.figure("Gauge map")

    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([75.5, 78.5, 30.25, 33.5])
    gl = ax.gridlines(draw_labels=True)
    divnorm = colors.TwoSlopeNorm(vmin=-500., vcenter=0, vmax=6000.)
    top_ds.elevation.plot.contourf(cmap='gist_earth', levels=np.arange(0, 6000, 100), norm=divnorm,
                                   cbar_kwargs={'label': 'Elevation [m]'})
    # for rec in bs_shape.records():
    #  ax.add_geometries([rec.geometry], cranfield_crs, edgecolor='None',
    #  facecolor="grey", alpha=0.2)
    ax.scatter(df['Longitude (o)'], df['Latitude (o)'], s=10,
               c='k', label="Gauge locations", zorder=9, linewidths=0.5)

    mask_filepath = 'Data/Masks/Beas_Sutlej_mask.nc'
    mask = xr.open_dataset(mask_filepath)
    mask = mask.rename({'latitude': 'nlat', 'longitude': 'nlon'})
    elv_da = top_ds.elevation.interp_like(mask)
    mask_da = mask.overlap
    masked_da = elv_da.where(mask_da > 0, drop=True)
    print(masked_da.median(skipna=True))

    # df_train = df[df['Longitude (o)'] > 77]
    # ax.scatter(df_train['Longitude (o)'],  df['Latitude (o)'], s=5, c='y', \
    # label='Test locations')
    # df_validation = df[df['Longitude (o)'] < 77]
    # ax.scatter(df_validation['Longitude (o)'],  df_validation['Latitude (o)'], \
    # s=5, c='b', label='Val locations')
    val_set = np.array([[31.65, 77.34], [31.424, 76.417], [31.80, 77.19],
                        [31.357, 76.878], [31.52, 77.22], [31.67, 77.06],
                        [31.454, 77.644], [31.77, 77.31], [31.238, 77.108],
                        [31.88, 77.15]])

    ax.scatter(val_set[:, 1], val_set[:, 0], s=10, c='r', edgecolors='k',
               label='Validation locations', linewidths=0.5, zorder=9)
    ax.scatter(hf_train_stations['lon'], hf_train_stations['lat'], s=10,
               c='dodgerblue', edgecolors='k', label='Training locations',
               linewidths=0.5, zorder=9)

    gl.top_labels = False
    gl.right_labels = False
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.add_feature(cf.BORDERS.with_scale("50m"), linewidth=0.5)

    plt.legend()
    plt.show()
