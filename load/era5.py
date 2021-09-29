"""
ERA5 reanalysis is downloaded via the Copernicus Data store.

The scripts use ERA5 re-analysis data which are available:
<https://cds.climate.copernicus.eu>.
You will need to create an account and a credentials file to use the API
included in the code (see <https://cds.climate.copernicus.eu/api-how-to>).
"""

import os
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import cdsapi

import location_sel as ls
from noaa_indices import indice_downloader


def collect_ERA5(location, minyear, maxyear):
    """ Downloads data from ERA5 for a given location"""
    era5_ds = download_data(location, xarray=True)
    if type(location) == str:
        loc_ds = ls.select_basin(era5_ds, location)
    else:
        lat, lon = location
        loc_ds = era5_ds.interp(
            coords={"lon": lon, "lat": lat}, method="nearest")
    tim_ds = loc_ds.sel(time=slice(minyear, maxyear))
    ds = tim_ds.assign_attrs(plot_legend="ERA5")  # in mm/day
    return ds


def gauge_download(station, minyear, maxyear):
    """Download and format ERA5 data for a given station name."""
    # Load data
    era5_beas_da = download_data('beas', xarray=True)
    beas_ds = era5_beas_da[['tp']]
    # Interpolate at location
    all_station_dict = pd.DataFrame.from_csv('Data/gauge_info.csv')
    lat, lon = all_station_dict[station]
    loc_ds = beas_ds.interp(coords={"lon": lon, "lat": lat}, method="nearest")
    tim_ds = loc_ds.sel(time=slice(minyear, maxyear))
    return tim_ds


def download_data(location, xarray=False, ensemble=False, all_var=False):
    """
    Downloads data for prepearation or analysis

    Inputs
        basin_filepath: string
        xarray: boolean
        ensemble: boolean
        all_var: boolean

    Returns
        df: DataFrame of data (!time not standardised!), or
        ds: DataArray of data
    """

    basin = ls.basin_finder(location)

    path = "Data/ERA5/"
    now = datetime.datetime.now()

    if ensemble is True:
        filename = "combi_data_ensemble" + "_" + \
            basin + "_" + now.strftime("%m-%Y") + ".csv"
    if all_var is True:
        filename = "all_data" + "_" + basin + \
            "_" + now.strftime("%m-%Y") + ".csv"
    elif ensemble is False:
        filename = "combi_data" + "_" + basin + \
            "_" + now.strftime("%m-%Y") + ".csv"

    filepath = path + filename
    print(filepath)

    if not os.path.exists(filepath):

        # Orography, humidity, precipitation and indices
        cds_df = cds_downloader(basin, ensemble=ensemble, all_var=all_var)
        ind_df = indice_downloader(all_var=all_var)
        df_combined = pd.merge_ordered(
            cds_df, ind_df, on="time", suffixes=("", "_y"))

        # Other variables not used in the GP
        if all_var is True:
            mean_df = mean_downloader(basin)
            uib_eofs_df = eof_downloader(basin, all_var=all_var)

            # Combine
            df_combined = pd.merge_ordered(df_combined, mean_df, on="time")
            df_combined = pd.merge_ordered(
                df_combined, uib_eofs_df, on=["time", "latitude", "longitude"]
            )

        # Choose experiment version 1
        expver1 = [c for c in df_combined.columns if c[-1] != '5']
        df_expver1 = df_combined[expver1]
        df_expver1.columns = df_expver1.columns.str.strip('_0001')

        # Pre pre-processing and save
        df_clean = df_expver1.dropna()  # .drop("expver", axis=1)
        df_clean['time'] = standardised_time(df_clean)
        df_clean["tp"] *= 1000  # to mm/day
        df_clean = df_clean.rename(
            columns={'latitude': 'lat', 'longitude': 'lon'})
        df_clean = df_clean.astype("float64")
        df_clean.to_csv(filepath)

        if xarray is True:
            if ensemble is True:
                df_multi = df_clean.set_index(
                    ["time", "long", "lat", "number"]
                )
            else:
                df_multi = df_clean.set_index(["time", "lon", "lat"])
            ds = df_multi.to_xarray()
            # Standardise time resolution
            maxyear = float(ds.time.max())
            minyear = float(ds.time.min())
            time_arr = np.arange(round(minyear) + 1./24., maxyear+0.05, 1./12.)
            print(ds)
            ds['time'] = time_arr
            return ds
        else:
            return df_clean

    else:
        df = pd.read_csv(filepath)
        df_clean = df.drop(columns=["Unnamed: 0"])

        if xarray is True:
            if ensemble is True:
                df_multi = df_clean.set_index(
                    ["time", "lon", "lat", "number"]
                )
            else:
                df_multi = df_clean.set_index(["time", "lon", "lat"])
            ds = df_multi.to_xarray()
            # Standardise time resolution
            maxyear = float(ds.time.max())
            minyear = float(ds.time.min())
            time_arr = np.arange(round(minyear) + 1./24.,
                                 maxyear + 0.05, 1./12.)
            ds['time'] = time_arr
            return ds
        else:
            return df_clean


def mean_downloader(basin):
    def mean_formatter(filepath, coords=None, name=None):
        """ Returns dataframe averaged data over a optionally given area """

        da = xr.open_dataset(filepath)

        if "expver" in list(da.dims):
            da = da.sel(expver=1)
            da = da.drop(["expver"])

        if coords is not None:
            da = da.sel(
                latitude=slice(coords[0], coords[2]),
                longitude=slice(coords[1], coords[3]),
            )

        mean_da = da.mean(dim=["longitude", "latitude"], skipna=True)
        clean_da = mean_da.assign_coords(
            time=(mean_da.time.astype("datetime64")))
        multiindex_df = clean_da.to_dataframe()
        df = multiindex_df  # .reset_index()
        if name is not None:
            df.rename(columns={"EOF": name}, inplace=True)

        return df

    # Temperature
    temp_filepath = update_cds_monthly_data(
        variables=["2m_temperature"], area=basin, qualifier="temp"
    )
    temp_df = mean_formatter(temp_filepath)

    # EOFs for 200hPa
    eof1_z200_c = mean_formatter(
        "Data/ERA5/regional_z200_EOF1.nc",
        coords=[40, 60, 35, 70], name="EOF200C1")
    eof1_z200_b = mean_formatter(
        "Data/ERA5/regional_z200_EOF1.nc",
        coords=[19, 83, 16, 93], name="EOF200B1")
    eof2_z200_c = mean_formatter(
        "Data/ERA5/regional_z200_EOF2.nc",
        coords=[40, 60, 35, 70], name="EOF200C2")
    eof2_z200_b = mean_formatter(
        "Data/ERA5/regional_z200_EOF2.nc",
        coords=[19, 83, 16, 93], name="EOF200B2")

    # EOFs for 500hPa
    eof1_z500_c = mean_formatter(
        "Data/ERA5/regional_z500_EOF1.nc",
        coords=[40, 60, 35, 70], name="EOF500C1")
    eof1_z500_b = mean_formatter(
        "Data/ERA5/regional_z500_EOF1.nc",
        coords=[19, 83, 16, 93], name="EOF500B1")
    eof2_z500_c = mean_formatter(
        "Data/ERA5/regional_z500_EOF2.nc",
        coords=[40, 60, 35, 70], name="EOF500C2")
    eof2_z500_b = mean_formatter(
        "Data/ERA5/regional_z500_EOF2.nc",
        coords=[19, 83, 16, 93], name="EOF500B2")

    # EOFs for 850hPa
    eof1_z850_c = mean_formatter(
        "Data/ERA5/regional_z850_EOF1.nc",
        coords=[40, 60, 35, 70], name="EOF850C1")
    eof1_z850_b = mean_formatter(
        "Data/ERA5/regional_z850_EOF1.nc",
        coords=[19, 83, 16, 93], name="EOF850B1")
    eof2_z850_c = mean_formatter(
        "Data/ERA5/regional_z850_EOF2.nc",
        coords=[40, 60, 35, 70], name="EOF850C2")
    eof2_z850_b = mean_formatter(
        "Data/ERA5/regional_z850_EOF2.nc",
        coords=[19, 83, 16, 93], name="EOF850B2")

    eof_df = pd.concat(
        [
            eof1_z200_b,
            eof1_z200_c,
            eof2_z200_b,
            eof2_z200_c,
            eof1_z500_b,
            eof1_z500_c,
            eof2_z500_b,
            eof2_z500_c,
            eof1_z850_b,
            eof1_z850_c,
            eof2_z850_b,
            eof2_z850_c,
        ],
        axis=1,
    )

    mean_df = pd.merge_ordered(temp_df, eof_df, on="time")

    return mean_df


def eof_downloader(basin, all_var=False):

    def eof_formatter(filepath, basin, name=None):
        """ Returns DataFrame of EOF over UIB  """

        da = xr.open_dataset(filepath)
        if "expver" in list(da.dims):
            da = da.sel(expver=1)
        (latmax, lonmin, latmin, lonmax) = ls.basin_extent(basin)
        sliced_da = da.sel(latitude=slice(latmin, latmax),
                           longitude=slice(lonmin, lonmax))

        eof_ds = sliced_da.EOF
        eof2 = eof_ds.assign_coords(time=(eof_ds.time.astype("datetime64")))
        eof_multiindex_df = eof2.to_dataframe()
        eof_df = eof_multiindex_df.dropna()
        eof_df.rename(columns={"EOF": name}, inplace=True)
        return eof_df

    # EOF UIB
    eof1_z200_u = eof_formatter(
        "Data/regional_z200_EOF1.nc", basin, name="EOF200U1"
    )
    eof1_z500_u = eof_formatter(
        "Data/regional_z500_EOF1.nc", basin, name="EOF500U1"
    )
    eof1_z850_u = eof_formatter(
        "Data/regional_z850_EOF1.nc", basin, name="EOF850U1"
    )

    eof2_z200_u = eof_formatter(
        "Data/regional_z200_EOF2.nc", basin, name="EOF200U2"
    )
    eof2_z500_u = eof_formatter(
        "Data/regional_z500_EOF2.nc", basin, name="EOF500U2"
    )
    eof2_z850_u = eof_formatter(
        "Data/regional_z850_EOF2.nc", basin, name="EOF850U2"
    )

    uib_eofs = pd.concat(
        [eof1_z200_u, eof2_z200_u, eof1_z500_u,
            eof2_z500_u, eof1_z850_u, eof2_z850_u],
        axis=1,
    )

    return uib_eofs


def cds_downloader(basin, ensemble=False, all_var=False):
    """ Return CDS Dataframe """

    if ensemble is False:
        cds_filepath = update_cds_monthly_data(area=basin)
    else:
        cds_filepath = update_cds_monthly_data(
            product_type="monthly_averaged_ensemble_members", area=basin)

    da = xr.open_dataset(cds_filepath)
    if "expver" in list(da.dims):
        da = da.sel(expver=1)

    multiindex_df = da.to_dataframe()
    cds_df = multiindex_df.reset_index()

    return cds_df


def standardised_time(dataset):
    """ Returns array of standardised times to plot """
    try:
        utime = dataset.time.values.astype(int)/(1e9 * 60 * 60 * 24 * 365)
    except Exception:
        time = np.array([d.strftime() for d in dataset.time.values])
        time2 = np.array([datetime.datetime.strptime(
            d, "%Y-%m-%d %H:%M:%S") for d in time])
        utime = np.array([d.timestamp() for d in time2]) / (60 * 60 * 24 * 365)
    return(utime + 1970)


def update_cds_monthly_data(
        dataset_name="reanalysis-era5-single-levels-monthly-means",
        product_type="monthly_averaged_reanalysis",
        variables=[
            "geopotential",
            "2m_dewpoint_temperature",
            "angle_of_sub_gridscale_orography",
            "slope_of_sub_gridscale_orography",
            "total_column_water_vapour",
            "total_precipitation",
        ],
        area=[40, 70, 30, 85],
        pressure_level=None,
        path="Data/ERA5/",
        qualifier=None):
    """
    Imports the most recent version of the given monthly ERA5 dataset as a
    netcdf from the CDS API.

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

    if qualifier is None:
        filename = (
            dataset_name + "_" + product_type +
            "_" + now.strftime("%m-%Y") + ".nc"
        )
    else:
        filename = (
            dataset_name
            + "_"
            + product_type
            + "_"
            + qualifier
            + "_"
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

        if pressure_level is None:
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
    Imports the most recent version of the given hourly ERA5 dataset as a
    netcdf from the CDS API.

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

    if qualifier is None:
        filename = (
            dataset_name + "_" + product_type +
            "_" + now.strftime("%m-%Y") + ".nc"
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

        if pressure_level is None:
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
