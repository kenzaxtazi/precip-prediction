# Data Downloader

import os
import glob
import calendar
import datetime
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import FileDownloader as fd

# Filepaths and URLs
# mask_filepath = 'Data/Masks/ERA5_Upper_Indus_mask.nc'


def download_data(mask_filepath, xarray=False, ensemble=False, all_var=False):
    """
    Downloads data for prepearation or analysis

    Inputs
        mask_filepath: string
        xarray: boolean
        ensemble: boolean
        all_var: boolean

    Returns
        df: DataFrame of data, or
        ds: DataArray of data
    """

    path = "Data/"
    now = datetime.datetime.now()

    if ensemble == True:
        filename = "combi_data_ensemble" + "_" + now.strftime("%m-%Y") + ".csv"
    if all_var == True:
        filename = "all_data" + "_" + now.strftime("%m-%Y") + ".csv"
    elif ensemble == False:
        filename = "combi_data" + "_" + now.strftime("%m-%Y") + ".csv"

    filepath = path + filename
    print(filepath)

    if not os.path.exists(filepath):

        # Orography, humidity, precipitation and indices
        cds_df = cds_downloader(mask_filepath, ensemble=ensemble, all_var=all_var)
        ind_df = indice_downloader(all_var=all_var)
        df_combined = pd.merge_ordered(cds_df, ind_df, on="time", suffixes=("", "_y"))

        # Other variables not used in the GP
        if all_var == True:
            mean_df = mean_downloader()
            uib_eofs_df = eof_downloader(mask_filepath, all_var=all_var)

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
        df_clean = df_expver1.dropna() #.drop("expver", axis=1)
        df_clean['time'] = standardised_time(df_clean)
        df_clean["tp"] *= 1000  # to mm/day
        df_clean = df_clean.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        df_clean = df_clean.astype("float64")
        df_clean.to_csv(filepath)

        if xarray == True:
            if ensemble == True:
                df_multi = df_clean.set_index(
                    ["time", "long", "lat", "number"]
                )
            else:
                df_multi = df_clean.set_index(["time", "lon", "lat"])
            ds = df_multi.to_xarray()
            return ds
        else:
            return df_clean

    else:
        df = pd.read_csv(filepath)
        df_clean = df.drop(columns=["Unnamed: 0"])

        if xarray == True:
            if ensemble == True:
                df_multi = df_clean.set_index(
                    ["time", "lon", "lat", "number"]
                )
            else:
                df_multi = df_clean.set_index(["time", "lon", "lat"])
            ds = df_multi.to_xarray()
            return ds
        else:
            return df_clean


def apply_mask(data_filepath, mask_filepath):

    """
    Opens NetCDF files and applies Upper Indus Basin mask to ERA 5 data.
    Inputs:
        Data filepath, NetCDF
        Mask filepath, NetCDF
    Return:
        A Data Array
    """
    da = xr.open_dataset(data_filepath)
    if "expver" in list(da.dims):
        print("expver found")
        da = da.sel(expver=1)

    mask = xr.open_dataset(mask_filepath)
    mask_da = mask.overlap

    # slice in case step has not been performed at download stage
    sliced_da = da.sel(latitude=slice(38, 30), longitude=slice(71.25, 82.75))

    UIB = sliced_da.where(mask_da > 0, drop=True)

    return UIB


def mean_downloader():
    def mean_formatter(filepath, coords=None, name=None):
        """ Returns dataframe averaged data over a optionally given area """

        da = xr.open_dataset(filepath)

        if "expver" in list(da.dims):
            da = da.sel(expver=1)
            da = da.drop(["expver"])

        if coords != None:
            da = da.sel(
                latitude=slice(coords[0], coords[2]),
                longitude=slice(coords[1], coords[3]),
            )

        mean_da = da.mean(dim=["longitude", "latitude"], skipna=True)
        clean_da = mean_da.assign_coords(time=(mean_da.time.astype("datetime64")))
        multiindex_df = clean_da.to_dataframe()
        df = multiindex_df  # .reset_index()
        if name != None:
            df.rename(columns={"EOF": name}, inplace=True)

        return df

    # Temperature
    temp_filepath = fd.update_cds_monthly_data(
        variables=["2m_temperature"], area=[40, 65, 20, 85], qualifier="temp"
    )
    temp_df = mean_formatter(temp_filepath)

    # EOFs for 200hPa
    eof1_z200_c = mean_formatter(
        "Data/regional_z200_EOF1.nc", coords=[40, 60, 35, 70], name="EOF200C1"
    )
    eof1_z200_b = mean_formatter(
        "Data/regional_z200_EOF1.nc", coords=[19, 83, 16, 93], name="EOF200B1"
    )
    eof2_z200_c = mean_formatter(
        "Data/regional_z200_EOF2.nc", coords=[40, 60, 35, 70], name="EOF200C2"
    )
    eof2_z200_b = mean_formatter(
        "Data/regional_z200_EOF2.nc", coords=[19, 83, 16, 93], name="EOF200B2"
    )

    # EOFs for 500hPa
    eof1_z500_c = mean_formatter(
        "Data/regional_z500_EOF1.nc", coords=[40, 60, 35, 70], name="EOF500C1"
    )
    eof1_z500_b = mean_formatter(
        "Data/regional_z500_EOF1.nc", coords=[19, 83, 16, 93], name="EOF500B1"
    )
    eof2_z500_c = mean_formatter(
        "Data/regional_z500_EOF2.nc", coords=[40, 60, 35, 70], name="EOF500C2"
    )
    eof2_z500_b = mean_formatter(
        "Data/regional_z500_EOF2.nc", coords=[19, 83, 16, 93], name="EOF500B2"
    )

    # EOFs for 850hPa
    eof1_z850_c = mean_formatter(
        "Data/regional_z850_EOF1.nc", coords=[40, 60, 35, 70], name="EOF850C1"
    )
    eof1_z850_b = mean_formatter(
        "Data/regional_z850_EOF1.nc", coords=[19, 83, 16, 93], name="EOF850B1"
    )
    eof2_z850_c = mean_formatter(
        "Data/regional_z850_EOF2.nc", coords=[40, 60, 35, 70], name="EOF850C2"
    )
    eof2_z850_b = mean_formatter(
        "Data/regional_z850_EOF2.nc", coords=[19, 83, 16, 93], name="EOF850B2"
    )

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


def eof_downloader(mask_filepath, all_var=False):
    def eof_formatter(filepath, mask_filepath, name=None):
        """ Returns DataFrame of EOF over UIB  """
        eof_da = apply_mask(filepath, mask_filepath)
        eof_ds = eof_da.EOF
        eof2 = eof_ds.assign_coords(time=(eof_ds.time.astype("datetime64")))
        eof_multiindex_df = eof2.to_dataframe()
        eof_df = eof_multiindex_df.dropna()
        eof_df.rename(columns={"EOF": name}, inplace=True)
        return eof_df

    # EOF UIB
    eof1_z200_u = eof_formatter(
        "Data/regional_z200_EOF1.nc", mask_filepath, name="EOF200U1"
    )
    eof1_z500_u = eof_formatter(
        "Data/regional_z500_EOF1.nc", mask_filepath, name="EOF500U1"
    )
    eof1_z850_u = eof_formatter(
        "Data/regional_z850_EOF1.nc", mask_filepath, name="EOF850U1"
    )

    eof2_z200_u = eof_formatter(
        "Data/regional_z200_EOF2.nc", mask_filepath, name="EOF200U2"
    )
    eof2_z500_u = eof_formatter(
        "Data/regional_z500_EOF2.nc", mask_filepath, name="EOF500U2"
    )
    eof2_z850_u = eof_formatter(
        "Data/regional_z850_EOF2.nc", mask_filepath, name="EOF850U2"
    )

    uib_eofs = pd.concat(
        [eof1_z200_u, eof2_z200_u, eof1_z500_u, eof2_z500_u, eof1_z850_u, eof2_z850_u],
        axis=1,
    )

    return uib_eofs


def indice_downloader(all_var=False):
    """ Return indice Dataframe """

    nao_url = "https://www.psl.noaa.gov/data/correlation/nao.data"
    n34_url = "https://psl.noaa.gov/data/correlation/nina34.data"
    n4_url = "https://psl.noaa.gov/data/correlation/nina4.data"

    n34_df = fd.update_url_data(n34_url, "N34")

    if all_var == False:
        ind_df = n34_df.astype("float64")
    else:
        nao_df = fd.update_url_data(nao_url, "NAO")
        n4_df = fd.update_url_data(n4_url, "N4")
        ind_df = n34_df.join([nao_df, n4_df])

    return ind_df


def cds_downloader(mask_filepath, ensemble=False, all_var=False):
    """ Return CDS Dataframe """

    if ensemble == False:
        cds_filepath = fd.update_cds_monthly_data()
    else:
        cds_filepath = fd.update_cds_monthly_data(
            product_type="monthly_averaged_ensemble_members"
        )

    masked_da = apply_mask(cds_filepath, mask_filepath)
    multiindex_df = masked_da.to_dataframe()
    cds_df = multiindex_df.reset_index()

    """
    if all_var == False:
        cds_df = cds_df.drop(['anor'], axis=1)
    """
    return cds_df


def standardised_time(dataset):
    """ Returns array of standardised times to plot """
    try:
        utime = dataset.time.values.astype(int)/(1e9 * 60 * 60 * 24 * 365)
    except Exception:
        time = np.array([d.strftime() for d in dataset.time.values])
        time2 = np.array([datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in time])
        utime = np.array([d.timestamp() for d in time2])/ ( 60 * 60 * 24 * 365)
    return(utime + 1970)


def collect_ERA5():
    """ Downloads data from ERA5 """
    mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"
    era5_ds= download_data(mask_filepath, xarray=True) # in m/day
    era5_ds = era5_ds.assign_attrs(plot_legend="ERA5")
    era5_ds = era5_ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    return era5_ds

def collect_CMIP5():
    """ Downloads data from CMIP5 """
    cmip_59_84_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_Amon_HadCM3_historical_r1i1p1_195912-198411.nc")
    cmip_84_05_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_Amon_HadCM3_historical_r1i1p1_198412-200512.nc")
    cmip_ds = cmip_84_05_ds.merge(cmip_59_84_ds)  # in kg/m2/s
    cmip_ds = cmip_ds.assign_attrs(plot_legend="HadCM3 historical")
    cmip_ds = cmip_ds.rename({'pr': 'tp'})
    cmip_ds['tp'] *= 60 * 60 * 24  # to mm/day
    cmip_ds['time'] = standardised_time(cmip_ds)
    return cmip_ds

def collect_CORDEX():
    """ Downloads data from CORDEX East Asia model """
    cordex_90_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_199001-199012.nc")
    cordex_91_00_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_199101-200012.nc")
    cordex_01_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_200101-201012.nc")
    cordex_02_11_ds = xr.open_dataset("/Users/kenzatazi/Downloads/pr_EAS-44i_ECMWF-ERAINT_evaluation_r1i1p1_MOHC-HadRM3P_v1_mon_201101-201111.nc")
    cordex_90_00_ds = cordex_90_ds.merge(cordex_91_00_ds)
    cordex_01_11_ds= cordex_01_ds.merge(cordex_02_11_ds)
    cordex_ds = cordex_01_11_ds.merge(cordex_90_00_ds)  # in kg/m2/s
    
    cordex_ds = cordex_ds.assign_attrs(plot_legend="CORDEX EA - MOHC-HadRM3P historical")
    cordex_ds = cordex_ds.rename_vars({'pr': 'tp'})
    cordex_ds['tp'] *= 60 * 60 * 24   # to mm/day
    cordex_ds['time'] = standardised_time(cordex_ds)

    return cordex_ds

def collect_APHRO():
    """ Downloads data from APHRODITE model"""
    aphro_ds = xr.merge([xr.open_dataset(f) for f in glob.glob('/Users/kenzatazi/Downloads/APHRO_MA_025deg_V1101.1951-2007.gz/*')])
    aphro_ds = aphro_ds.assign_attrs(plot_legend="APHRODITE") # in mm/day   
    aphro_ds = aphro_ds.rename_vars({'precip': 'tp'})
    aphro_ds['time'] = standardised_time(aphro_ds)
    return aphro_ds

def collect_CRU():
    """ Downloads data from CRU model"""
    cru_ds = xr.open_dataset("/Users/kenzatazi/Downloads/cru_ts4.04.1901.2019.pre.dat.nc")
    cru_ds = cru_ds.assign_attrs(plot_legend="CRU") # in mm/month
    cru_ds = cru_ds.rename_vars({'pre': 'tp'})
    cru_ds['tp'] /= 30.437  #TODO apply proper function to get mm/day
    cru_ds['time'] = standardised_time(cru_ds)
    return cru_ds