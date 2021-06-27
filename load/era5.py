"""
ERA5 reanalysis is downloaded via the Copernicus Data store.

Variables
    Total precipitation (tp) in m/day
"""

import calendar
import datetime
import os
import urllib
import numpy as np
import xarray as xr
import pandas as pd
import ftplib
import cdsapi

import DataDownloader as dd
import LocationSel as ls


def update_cds_monthly_data(
    dataset_name="reanalysis-era5-single-levels-monthly-means",
    product_type="monthly_averaged_reanalysis",
    variables=[
        "2m_dewpoint_temperature",
        "angle_of_sub_gridscale_orography",
        "orography",
        "slope_of_sub_gridscale_orography",
        "total_column_water_vapour",
        "total_precipitation",
    ],
    area=[40, 70, 30, 85],
    pressure_level=None,
    path="Data/ERA5/",
    qualifier=None):
    """
    Imports the most recent version of the given monthly ERA5 dataset as a netcdf from the CDS API.

    Inputs:
        dataset_name: str
        prduct_type: str
        variables: list of strings
        pressure_level: str or None
        area: list of scalars
        path: str
        qualifier: str

    Returns: local filepath to netcdf.
    """
    if type(area) == str:
        qualifier = area
        area = ls.basin_extent(area)
        
    now = datetime.datetime.now()

    if qualifier == None:
        filename = (
            dataset_name + "_" + product_type + "_" + now.strftime("%m-%Y") + ".nc"
        )
    else:
        filename = (
            dataset_name
            + "_"
            + product_type
            + "_"
            + qualifier
            +"_"
            + now.strftime("%m-%Y")
            + ".nc"
        )

    filepath = path + filename

    # Only download if updated file is not present locally
    if not os.path.exists(filepath):

        current_year = now.strftime("%Y")
        years = np.arange(1979, int(current_year) + 1, 1).astype(str)
        months = np.arange(1, 13, 1).astype(str)

        c = cdsapi.Client()

        if pressure_level == None:
            c.retrieve(
                "reanalysis-era5-single-levels-monthly-means",
                {
                    "format": "netcdf",
                    "product_type": product_type,
                    "variable": variables,
                    "year": years.tolist(),
                    "time": "00:00",
                    "month": months.tolist(),
                    "area": area,
                },
                filepath,
            )
        else:
            c.retrieve(
                "reanalysis-era5-single-levels-monthly-means",
                {
                    "format": "netcdf",
                    "product_type": product_type,
                    "variable": variables,
                    "pressure_level": pressure_level,
                    "year": years.tolist(),
                    "time": "00:00",
                    "month": months.tolist(),
                    "area": area,
                },
                filepath,
            )

    return filepath


def update_cds_hourly_data(
    dataset_name="reanalysis-era5-pressure-levels",
    product_type="reanalysis",
    variables=["geopotential"],
    pressure_level="200",
    area=[90, -180, -90, 180],
    path="Data/ERA5/",
    qualifier=None):
    """
    Imports the most recent version of the given hourly ERA5 dataset as a netcdf from the CDS API.

    Inputs:
        dataset_name: str
        prduct_type: str
        variables: list of strings
        area: list of scalars
        pressure_level: str or None
        path: str
        qualifier: str

    Returns: local filepath to netcdf.
    """
    now = datetime.datetime.now()

    if qualifier == None:
        filename = (
            dataset_name + "_" + product_type + "_" + now.strftime("%m-%Y") + ".nc"
        )
    else:
        filename = (
            dataset_name
            + "_"
            + product_type
            + "_"
            + now.strftime("%m-%Y")
            + "_"
            + qualifier
            + ".nc"
        )

    filepath = path + filename

    # Only download if updated file is not present locally
    if not os.path.exists(filepath):

        current_year = now.strftime("%Y")
        years = np.arange(1979, int(current_year) + 1, 1).astype(str)
        months = np.arange(1, 13, 1).astype(str)
        days = np.arange(1, 32, 1).astype(str)

        c = cdsapi.Client()

        if pressure_level == None:
            c.retrieve(
                dataset_name,
                {
                    "format": "netcdf",
                    "product_type": product_type,
                    "variable": variables,
                    "year": years.tolist(),
                    "time": "00:00",
                    "month": months.tolist(),
                    "day": days.tolist(),
                    "area": area,
                },
                filepath,
            )
        else:
            c.retrieve(
                dataset_name,
                {
                    "format": "netcdf",
                    "product_type": product_type,
                    "variable": variables,
                    "pressure_level": pressure_level,
                    "year": years.tolist(),
                    "time": "00:00",
                    "month": months.tolist(),
                    "day": days.tolist(),
                    "area": area,
                },
                filepath,
            )

    return filepath



def collect_ERA5(location, minyear, maxyear):
    """ Downloads data from ERA5 for a given location"""
    era5_ds= dd.download_data(location, xarray=True) 

    if type(location) == str:
        loc_ds = ls.select_basin(era5_ds, location)
    else:
        lat, lon = location
        loc_ds = era5_ds.interp(coords={"lon": lon, "lat": lat}, method="nearest")

    tim_ds = loc_ds.sel(time= slice(minyear, maxyear))
    ds = tim_ds.assign_attrs(plot_legend="ERA5") # in mm/day   
    return ds


def gauge_download(station, minyear, maxyear):
    """ 
    Download and format ERA5 data 
    
    Args:
        station (str): station name (capitalised)
    
    Returns
       ds (xr.DataSet): ERA values for precipitation
    """

    station_dict = {'Banjar': [31.65,77.34], 'Sainj': [31.77,76.933], 'Ganguwal': [31.25,76.486]}
    
    # Load data
    era5_beas_da = dd.download_data('beas', xarray=True)
    beas_ds = era5_beas_da[['tp']]
    
    # Interpolate at location
    lat, lon = station_dict[station]
    loc_ds = beas_ds.interp(coords={"lon": lon, "lat": lat}, method="nearest")
    tim_ds = loc_ds.sel(time= slice(minyear, maxyear))

    return tim_ds