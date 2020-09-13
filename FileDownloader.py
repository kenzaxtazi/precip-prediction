# File Downloader

import calendar
import datetime
import os
import urllib
import numpy as np
import xarray as xr
import pandas as pd
import ftplib
import cdsapi


def save_csv_from_url(url, saving_path):
    """ Downloads data from a url and saves it to a specified path. """
    response = urllib.request.urlopen(url)
    with open(saving_path, "wb") as f:
        f.write(response.read())


def update_url_data(url, name):
    """ Import the most recent dataset from URL and return it as pandas DataFrame """

    filepath = "Data/"
    now = datetime.datetime.now()
    file = filepath + name + "-" + now.strftime("%m-%Y") + ".csv"

    # Only download CSV if not present locally
    if not os.path.exists(file):
        save_csv_from_url(url, file)

    # create and format DataFrame
    df = pd.read_csv(file)
    df_split = df[list(df)[0]].str.split(expand=True)
    df_long = pd.melt(
        df_split,
        id_vars=[0],
        value_vars=np.arange(1, 13),
        var_name="month",
        value_name=name,
    )

    # Create a datetime column
    df_long["time"] = df_long[0].astype(str) + "-" + df_long["month"].astype(str)
    df_long["time"] = pd.to_datetime(df_long["time"], errors="coerce")

    # Clean
    df_clean = df_long.dropna()
    df_sorted = df_clean.sort_values(by=["time"])
    df_final = df_sorted.set_index("time")

    return pd.DataFrame(df_final[name])


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
    pressure_level=None,
    area=[40, 70, 30, 85],
    path="Data/",
    qualifier=None,
):
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
    path="Data/",
    qualifier=None,
):
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
