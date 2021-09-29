# Trend comparison

import os
import numpy as np
from numpy.ma.core import append
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import cftime
from scipy import stats
from tqdm import tqdm

from load import beas_sutlej_gauges, era5, cru, beas_sutlej_wrf, gpm, aphrodite


import DataDownloader as dd
import DataPreparation as dp
import GPModels as gp
import Correlation as corr
import Timeseries as tims
import PDF as pdf
import Maps

from sklearn.metrics import mean_squared_error, r2_score


model_filepath = 'Models/model_2021-01-01/08/21-22-35-56'


def model_prep(location, data_filepath='Data/model_pred_test.csv', model_filepath='Models/model_2021-01-01/08/21-22-35-56', minyear=1990, maxyear=2005, ):
    """ Prepares model outputs for comparison """

    if os.path.exists(data_filepath):
        model_df = pd.read_csv(data_filepath)

    else:
        model = gp.restore_model(model_filepath)

        xtr, _ytr = dp.areal_model_eval(
            location, minyear=minyear, maxyear=maxyear)
        y_gpr, y_std = model.predict_y(xtr)

        # to mm/day
        y_gpr_t = dp.inverse_log_transform(y_gpr) * 1000
        y_std_t = dp.inverse_log_transform(y_std) * 1000

        model_df = pd.DataFrame({'time': xtr[:, 0]+1970,
                                'lat': xtr[:, 1],
                                 'lon': xtr[:, 2],
                                 'tp': y_gpr_t.flatten(),
                                 'tp_std': y_std_t.flatten()})

        model_df = model_df.groupby('time').mean()
        model_df.to_csv(data_filepath)

    reset_df = model_df.reset_index()
    multi_index_df = reset_df.set_index(["time", "lat", "lon"])

    model_ds = multi_index_df.to_xarray()
    model_ds = model_ds.assign_attrs(plot_legend="Model")
    return model_ds


def dataset_stats(datasets, ref_ds=None, ret=False):
    """ Print mean, standard deviations and slope for datasets """

    r2_list = []
    rmse_list = []

    for ds in datasets:

        name = ds.plot_legend
        tp = ds.tp.values
        if len(tp.shape) > 1:
            ds = dp.average_over_coords(ds)
        da = ds['tp'].dropna(dim='time')

        slope, _intercept, _r_value, _p_value, _std_err = stats.linregress(
            da.time.values, da.values)
        print(name)
        print('mean = ', np.mean(da.values), 'mm/day')
        print('std = ', np.std(da.values), 'mm/day')
        print('slope = ', slope, 'mm/day/year')

        if ref_ds is not None:
            tp_ref = ref_ds.tp.values
            df = pd.DataFrame({'tp_ref': tp_ref, 'tp': tp})
            df = df.dropna()
            r2 = r2_score(df['tp_ref'].values, df['tp'].values)
            rmse = mean_squared_error(
                df['tp_ref'].values, df['tp'].values, squared=False)
            print('R2 = ', r2)
            print('RMSE = ', rmse)
            r2_list.append(r2)
            rmse_list.append(rmse)

    if ret == True:
        return [r2_list, rmse_list]


def single_location_comparison(location=[31.65, 77.34], station='Banjar', min_year=2000, max_year=2011):
    """ Plots model outputs for given coordinates over time """

    aphro_ds = aphrodite.collect_APHRO(
        location, minyear=min_year, maxyear=max_year)
    cru_ds = cru.collect_CRU(location, minyear=min_year, maxyear=max_year)
    era5_ds = era5.collect_ERA5(location, minyear=min_year, maxyear=max_year)
    gpm_ds = gpm.collect_GPM(location,  minyear=min_year, maxyear=max_year)
    wrf_ds = beas_sutlej_wrf.collect_BC_WRF(
        location, minyear=min_year, maxyear=max_year)
    gauge_ds = beas_sutlej_gauges.gauge_download(
        station, minyear=min_year, maxyear=max_year)

    # cmip_ds = dd.collect_CMIP5()
    # cordex_ds = dd.collect_CORDEX()
    # model_ts = model_prep([lat, lon], data_filepath='single_loc_test.csv', model_filepath=model_filepath)

    timeseries = [gauge_ds, gpm_ds, era5_ds, wrf_ds, aphro_ds, cru_ds]

    tims.benchmarking_subplots(timeseries, reference_dataset=gauge_ds)
    dataset_stats(timeseries, ref_ds=gauge_ds)
    # corr.dataset_correlation(timeseries)
    # pdf.benchmarking_plot(timeseries, kernel_density=False)


def basin_comparison(model_filepath, location):
    """ Plots model outputs for given coordinates over time """

    aphro_ds = aphrodite.collect_APHRO(location, minyear=2000, maxyear=2011)
    cru_ds = cru.collect_CRU(location, minyear=2000, maxyear=2011)
    era5_ds = era5.collect_ERA5(location, minyear=2000, maxyear=2011)
    gpm_ds = gpm.collect_GPM(location,  minyear=2000, maxyear=2011)
    wrf_ds = beas_sutlej_wrf.collect_BC_WRF(
        location, minyear=2000, maxyear=2011)

    # cmip_ds = dd.collect_CMIP5()
    # cordex_ds = dd.collect_CORDEX()
    # cmip_bs = select_basin(cmip_ds, location)
    # cordex_bs = select_basin(cordex_ds, location)
    # model_bs = model_prep(location, model_filepath)

    basins = [aphro_ds, cru_ds, era5_ds, gpm_ds, wrf_ds]

    dataset_stats(basins)


def multi_location_comparison():
    """ Plots model outputs for given coordinates over time """

    gauge_ds = beas_sutlej_gauges.all_gauge_data(
        minyear=2000, maxyear=2011, threshold=3653)
    locations = [[31.424, 76.417], [31.357, 76.878], [31.52, 77.22], [31.67, 77.06], [31.454, 77.644],
                 [31.238, 77.108], [31.65, 77.34], [31.88, 77.15], [31.77, 77.31], [31.80, 77.19]]

    aphro_sets = []
    cru_sets = []
    era5_sets = []
    gpm_sets = []
    wrf_sets = []

    for l in locations:
        aphro_ds = aphrodite.collect_APHRO(l, minyear=2000, maxyear=2011)
        aphro_sets.append(aphro_ds.tp)

        cru_ds = cru.collect_CRU(l, minyear=2000, maxyear=2011)
        cru_sets.append(cru_ds.tp)

        era5_ds = era5.collect_ERA5(l, minyear=2000, maxyear=2011)
        era5_sets.append(era5_ds.tp)

        gpm_ds = gpm.collect_GPM(l,  minyear=2000, maxyear=2011)
        gpm_sets.append(gpm_ds.tp)

        wrf_ds = beas_sutlej_wrf.collect_BC_WRF(l, minyear=2000, maxyear=2011)
        wrf_sets.append(wrf_ds.tp)

    # Merge datasets
    aphro_mer_ds = xr.concat(aphro_sets, dim='lon')
    aphro_mer_ds.attrs['plot_legend'] = 'APHRODITE'

    cru_mer_ds = xr.concat(cru_sets, dim='lon')
    cru_mer_ds.attrs['plot_legend'] = 'CRU'

    era5_mer_ds = xr.concat(era5_sets, dim='lon')
    era5_mer_ds.attrs['plot_legend'] = 'ERA5'

    gpm_mer_ds = xr.concat(gpm_sets, dim='lon')
    gpm_mer_ds.attrs['plot_legend'] = 'TRMM'

    wrf_mer_ds = xr.concat(wrf_sets, dim='lon')
    wrf_mer_ds.attrs['plot_legend'] = 'BC WRF'

    timeseries = [gpm_mer_ds, era5_mer_ds,
                  wrf_mer_ds, aphro_mer_ds, cru_mer_ds]
    pdf.mult_gauge_loc_plot(gauge_ds, timeseries)


def gauge_stats():
    """ Print mean, standard deviations and slope for datasets """

    bs_station_dict = {'Arki': [31.154, 76.964], 'Banjar': [31.65, 77.34], 'Banjar IMD': [31.637, 77.344],
                       'Berthin': [31.471, 76.622], 'Bhakra': [31.424, 76.417], 'Barantargh': [31.087, 76.608],
                       'Bhoranj': [31.648, 76.698], 'Bhuntar': [31.88, 77.15], 'Daslehra': [31.4, 76.55],
                       'Dehra': [31.885, 76.218], 'Ganguwal': [31.25, 76.486], 'Ghanauli': [30.994, 76.527],
                       'Ghumarwin': [31.436, 76.708], 'Hamirpur': [31.684, 76.519], 'Janjehl': [31.52, 77.22],
                       'Jogindernagar': [32.036, 76.734], 'Kalatop': [32.552, 76.018], 'Kalpa': [31.54, 78.258],
                       'Kangra': [32.103, 76.271], 'Karsog': [31.383, 77.2], 'Kasol': [31.357, 76.878],
                       'Kaza': [32.225, 78.072], 'Kotata': [31.233, 76.534], 'Kumarsain': [31.317, 77.45],
                       'Larji': [31.80, 77.19], 'Lohard': [31.204, 76.561], 'Mashobra': [31.13, 77.229],
                       'Nadaun': [31.783, 76.35], 'Naina Devi': [31.279, 76.554], 'Nangal': [31.368, 76.404],
                       'Olinda': [31.401, 76.385], 'Palampur': [32.107, 76.543], 'Pandoh': [31.67, 77.06],
                       'Rampur': [31.454, 77.644], 'Rampur IMD': [31.452, 77.633], 'Sadar-Bilarspur': [31.348, 76.762],
                       'Sadar-Mandi': [31.712, 76.933], 'Sainj': [31.77, 76.933], 'Salooni': [32.728, 76.034],
                       'Sarkaghat': [31.704, 76.812], 'Sujanpur': [31.832, 76.503], 'Sundernargar': [31.534, 76.905],
                       'Suni': [31.238, 77.108], 'Suni IMD': [31.23, 77.164], 'Swaghat': [31.713, 76.746],
                       'Theog': [31.124, 77.347]}

    mlm_val_stations = {'Bhakra': [31.424, 76.417], 'Suni': [31.238, 77.108], 'Pandoh': [31.67, 77.06],
                        'Janjehl': [31.52, 77.22], 'Bhuntar': [31.88, 77.15], 'Rampur': [31.454, 77.644]}

    val_stations = ['Banjar', 'Larji', 'Bhuntar', 'Sainj',
                    'Bhakra', 'Kasol', 'Suni', 'Pandoh', 'Janjehl', 'Rampur']

    r2_list = []
    rmse_list = []

    for s in tqdm(bs_station_dict):

        gauge_ds = beas_sutlej_gauges.gauge_download(
            s, minyear=2000, maxyear=2011)
        gauge_maxy = gauge_ds.time.max().values
        gauge_miny = gauge_ds.time.min().values
        miny = gauge_miny - 0.0001
        maxy = gauge_maxy + 0.0001

        location = bs_station_dict[s]

        aphro_ds = aphrodite.collect_APHRO(
            location, minyear=miny, maxyear=maxy)
        cru_ds = cru.collect_CRU(location, minyear=miny, maxyear=maxy)
        era5_ds = era5.collect_ERA5(location, minyear=miny, maxyear=maxy)
        gpm_ds = gpm.collect_GPM(location,  minyear=miny, maxyear=maxy)
        wrf_ds = beas_sutlej_wrf.collect_BC_WRF(
            location, minyear=miny, maxyear=maxy)

        timeseries = [era5_ds, gpm_ds, aphro_ds, cru_ds, wrf_ds]
        r2s, rmses = dataset_stats(timeseries, ref_ds=gauge_ds, ret=True)
        r2_list.append(r2s)
        rmse_list.append(rmses)

    avg_r2 = np.array(r2_list).mean(axis=0)
    avg_rmse = np.array(rmse_list).mean(axis=0)

    return avg_r2, avg_rmse
