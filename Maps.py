# Collection of map plotting functions

import calendar

import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.cm as cm
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.colors as colors

from shapely.geometry import Polygon, shape, LinearRing
from cartopy.io import shapereader
from cartopy import config
from scipy import signal

import DataDownloader as dd
from load import era5, cru, beas_sutlej_wrf, gpm, aphrodite

#data_filepath = fd.update_cds_monthly_data()
# mask_filepath = "Data/Masks/ERA5_Upper_Indus_mask.nc"

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
        ax.gridlines(draw_labels=True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.show()


def global_map(uib_only=False, bs_only=False):
    """ 
    Returns global map with study area(s) superimposed.
    """
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

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree(central_longitude=70))
    ax.coastlines("50m", linewidth=0.5)

    if bs_only == False:
        for rec in uib_shape.records():
            ax.add_geometries(
                [rec.geometry],
                ccrs.AlbersEqualArea(
                    central_longitude=125, central_latitude=-15, standard_parallels=(7, -32)
                ),
                edgecolor=None,
                facecolor="red",
                alpha=0.1)

    if uib_only == False:
        for rec in bs_shape.records():
            ax.add_geometries(
                [rec.geometry],
                cranfield_crs,
                edgecolor=None,
                facecolor="green",
                alpha=0.1)

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


def multi_dataset_map(location, seasonal=False):
    """ 
    Create maps of raw values from multiple datasets
    - ERA5
    - GPM PR
    - APHRODITE
    - CRU
    - Bannister corrected WRF
    """

    # Load data 
    aphro_ds = aphrodite.collect_APHRO(location, minyear=2000, maxyear=2011)
    cru_ds= cru.collect_CRU(location, minyear=2000, maxyear=2011)
    era5_ds = era5.collect_ERA5(location, minyear=2000, maxyear=2011)
    gpm_ds = gpm.collect_GPM(location,  minyear=2000, maxyear=2011)
    wrf_ds = beas_sutlej_wrf.collect_BC_WRF(location, minyear=2000, maxyear=2011)

    dataset_list = [gpm_ds, era5_ds, wrf_ds, aphro_ds, cru_ds]

    # Slice and take averages
    avg_list = []
    for ds in dataset_list: 
        ds_avg =  ds.tp.mean(dim='time')

        if seasonal == True:
            ds_annual_avg = ds_avg
            
            ds_jun = ds.tp[5::12]
            ds_jul = ds.tp[6::12]
            ds_aug = ds.tp[7::12]
            ds_sep = ds.tp[8::12]
            ds_monsoon = xr.merge([ds_jun, ds_jul, ds_aug, ds_sep])
            ds_monsoon_avg = ds_monsoon.tp.mean(dim='time')

            ds_dec = ds.tp[11::12]
            ds_jan = ds.tp[0::12]
            ds_feb = ds.tp[1::12]
            ds_mar = ds.tp[2::12]
            ds_west = xr.merge([ds_dec, ds_jan, ds_feb, ds_mar])
            ds_west_avg = ds_west.tp.mean(dim='time')

            ds_avg = xr.concat([ds_annual_avg, ds_monsoon_avg, ds_west_avg], pd.Index(["Annual", "Monsoon (JJAS)", "Winter (DJFM)"], name='t'))

        avg_list.append(ds_avg)

    datasets = xr.concat(avg_list, pd.Index(["TRMM", "ERA5", "BC_WRF", "APHRO", "CRU" ], name="Dataset")) 
        
    # Plot maps

    g = datasets.plot(
        x="lon",
        y="lat",
        col="t",
        row= "Dataset",
        cbar_kwargs={"label": "Total precipitation (mm/day)"},
        cmap="YlGnBu",
        subplot_kws={"projection": ccrs.PlateCarree()})

    for ax in g.axes.flat:
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.set_extent([75, 83.5, 29, 34])
        ax.add_feature(cf.BORDERS)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.show()


def beas_sutlej_gauge_map(): # TODO could include length of datasets as marker size
    """ Maps of gauges used by Bannister """

    station_dict = {'Arki':[31.154, 76.964, 1176], 'Banjar': [31.65, 77.34, 1914], 'Banjar IMD':[31.637, 77.344, 1427],  
                'Berthin':[31.471, 76.622, 657], 'Bhakra':[31.424, 76.417, 518], 'Barantargh': [31.087, 76.608, 285], 
                'Bharmaur': [32.45, 76.533, 1867], 'Bhoranj':[31.648, 76.698, 834], 'Bhuntar': [31.88, 77.15, 1100], 
                'Churah': [32.833, 76.167, 1358], 'Dadahu':[30.599, 77.437, 635], 'Daslehra': [31.4, 76.55, 561], 
                'Dehra': [31.885, 76.218, 472], 'Dhaula Kuan': [30.517, 77.479, 443], 'Ganguwal': [31.25, 76.486, 345], 
                'Ghanauli': [30.994, 76.527, 284], 'Ghumarwin': [31.436, 76.708, 640], 'Hamirpur': [31.684, 76.519, 763], 
                'Janjehl': [31.52, 77.22, 2071], 'Jogindernagar': [32.036, 76.734, 1442], 'Jubbal':[31.12, 77.67, 2135], 
                'Kalatop': [32.552, 76.018, 2376], 'Kalpa': [31.54, 78.258, 2439], 'Kandaghat': [30.965, 77.119, 1339], 
                'Kangra': [32.103, 76.271, 1318], 'Karsog': [31.383, 77.2, 1417], 'Kasol': [31.357, 76.878, 662], 
                'Kaza': [32.225, 78.072, 3639], 'Kotata': [31.233, 76.534, 320], 'Kothai': [31.119, 77.485, 1531],
                'Kumarsain': [31.317, 77.45, 1617], 'Larji': [31.80, 77.19, 975], 'Lohard': [31.204, 76.561, 290], 
                'Mashobra': [31.13, 77.229, 2240], 'Nadaun': [31.783, 76.35, 480], 'Nahan': [30.559, 77.289, 874], 
                'Naina Devi': [31.279, 76.554, 680], 'Nangal': [31.368, 76.404, 354], 'Olinda': [31.401, 76.385, 363],
                'Pachhad': [30.777, 77.164, 1257], 'Palampur': [32.107, 76.543, 1281], 'Pandoh':[31.67,77.06, 899], 
                'Paonta Sahib': [30.47, 77.625, 433], 'Rakuna': [30.605, 77.473, 688], 'Rampur': [31.454,77.644, 976],
                'Rampur IMD': [31.452, 77.633, 972], 'Rohru':[31.204, 77.751, 1565], 'Sadar-Bilarspur':[31.348, 76.762, 576], 
                'Sadar-Mandi': [31.712, 76.933, 761], 'Sainj': [31.77, 77.31, 1280] , 'Salooni':[32.728, 76.034, 1785],
                'Sarkaghat': [31.704, 76.812, 1155], 'Sujanpur':[31.832, 76.503, 557], 'Sundernargar': [31.534, 76.905, 889], 
                'Suni':[31.238,77.108, 655], 'Suni IMD':[31.23, 77.164, 765], 'Swaghat': [31.713, 76.746, 991], 
                'Theog': [31.124, 77.347, 2101]}

    station_df = pd.DataFrame.from_dict(station_dict, orient='index',columns=['lat', 'lon', 'elv'])
    station_df = station_df.reset_index()

    hf_train_df1 = station_df[(station_df['lon']< 77.0) & (station_df['lat']> 32)]
    hf_train_df2 = station_df[(station_df['lon']< 76.60) & ((station_df['lat']< 32) & (station_df['lat']> 31.6))]
    hf_train_df3 = station_df[(station_df['lon']> 77.0) & (station_df['lat']< 31)]
    hf_train_df4 = station_df[(station_df['lon']< 78.0) & (station_df['lon']> 77.0) & (station_df['lat']> 31) & (station_df['lat']< 31.23)]
    hf_train_df5 = station_df[(station_df['lon']> 78.2)]
    hf_train_stations = pd.concat([hf_train_df1, hf_train_df2, hf_train_df3, hf_train_df4, hf_train_df5])

    # Gauges
    filepath= 'Data/stations_MGM.xlsx'
    df = pd.read_excel(filepath)

    # Beas and Sutlej shapefile and projection
    bs_path = "Data/Shapefiles/beas-sutlej-shapefile/12500Ha.shp"
    bs_shape = shapereader.Reader(bs_path)
    bs_globe = ccrs.Globe(semimajor_axis=6377276.345, inverse_flattening=300.8017)
    cranfield_crs = ccrs.LambertConformal(
                central_longitude=82, central_latitude=20, false_easting=2000000.0, false_northing=2000000.0,
                standard_parallels=[12.47294444444444,35.17280555555556], globe=bs_globe)

    # Topography
    top_ds = xr.open_dataset('/Users/kenzatazi/Downloads/GMTED2010_15n015_00625deg.nc')
    top_ds = top_ds.assign_coords({'nlat': top_ds.latitude, 'nlon': top_ds.longitude})
    top_ds = top_ds.sel(nlat=slice(29,34), nlon=slice(75, 83))

    # Figure 
    plt.figure("Gauge map")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([75, 83, 29, 34])
 
    divnorm = colors.TwoSlopeNorm(vmin=-500., vcenter=0, vmax=6000.)
    top_ds.elevation.plot.contourf(cmap='gist_earth', levels=np.arange(0, 6000, 100), norm=divnorm,
                                   cbar_kwargs={'label': 'Elavation [m]'})
    for rec in bs_shape.records():
        ax.add_geometries([rec.geometry], cranfield_crs, edgecolor='None', facecolor="blue", alpha=0.2)
    ax.scatter(df['Longitude (o)'], df['Latitude (o)'], s=5, c='k', label="Gauge locations")

    mask_filepath = 'Data/Masks/Beas_Sutlej_mask.nc'
    mask = xr.open_dataset(mask_filepath)
    mask = mask.rename({'latitude': 'nlat', 'longitude': 'nlon'})
    elv_da = top_ds.elevation.interp_like(mask)
    mask_da = mask.overlap
    masked_da = elv_da.where(mask_da > 0, drop=True)
    print(masked_da.median(skipna=True))

    # df_train = df[df['Longitude (o)'] > 77]
    # ax.scatter(df_train['Longitude (o)'],  df['Latitude (o)'], s=5, c='y', label='Test locations')
    # df_validation = df[df['Longitude (o)'] < 77]
    # ax.scatter(df_validation['Longitude (o)'],  df_validation['Latitude (o)'], s=5, c='b', label='Val locations')
    val_set = np.array([[31.65, 77.34], [31.424, 76.417], [31.80, 77.19], [31.357, 76.878], [31.52, 77.22], 
                 [31.67, 77.06], [31.454, 77.644], [31.77, 77.31], [31.238,77.108], [31.88, 77.15]])

    ax.scatter(val_set[:,1], val_set[:,0], s=5, c='r', label='Validation locations')
    ax.scatter(hf_train_stations['lon'], hf_train_stations['lat'], s=5, c='deepskyblue', label='Training locations')
    
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.add_feature(cf.BORDERS.with_scale("50m"), linewidth=0.5)
    
    plt.legend()
    plt.show()

     