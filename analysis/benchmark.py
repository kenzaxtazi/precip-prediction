# Trend comparison

import os
import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')  # noqa
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/precip-prediction')  # noqa

import gp.data_prep as dp
#import gp.gp_models as gp
import analysis.pdf as pdf
import analysis.timeseries as tims
from load import beas_sutlej_gauges, era5, cru, beas_sutlej_wrf, gpm, aphrodite, data_dir
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from scipy import stats
import xarray as xr
import numpy as np
import pandas as pd


model_filepath = 'Models/model_2021-01-01/08/21-22-35-56'


def model_prep(location, data_filepath='_Data/model_pred_test.csv',
               model_filepath='_Models/model_2021-01-01/08/21-22-35-56',
               minyear=1990, maxyear=2005):
    """Prepare model outputs for comparison."""

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
    """Print mean, standard deviations and slope for datasets."""

    r2_list = []
    rmse_list = []
    rmse_p5_list = []
    rmse_p95_list = []
    r2_p5_list = []
    r2_p95_list = []

    for ds in datasets:

        name = ds.plot_legend
        tp = ds.tp.values
        if len(tp.shape) > 1:
            ds = dp.average_over_coords(ds)
        da = ds['tp'].dropna(dim='time')
        '''
        slope, _intercept, _r_value, _p_value, _std_err = stats.linregress(
            da.time.values, da.values)

        print(name)
        print('mean = ', np.mean(da.values), 'mm/day')
        print('std = ', np.std(da.values), 'mm/day')
        print('slope = ', slope, 'mm/day/year')
        '''
        if ref_ds is not None:
            tp_ref = ref_ds.tp.values
            print(tp_ref.shape, tp.shape)
            df = pd.DataFrame({'tp_ref': tp_ref, 'tp': tp})
            df = df.dropna()

            y_true = df['tp_ref'].values
            y_pred = df['tp'].values

            # all values
            r2 = r2_score(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)

            # 5th percentile
            p5 = np.percentile(y_true, 5.0)
            indx = [y_true <= p5][0]
            y_true_p5 = y_true[indx]
            y_pred_p5 = y_pred[indx]
            r2_p5 = r2_score(y_true_p5, y_pred_p5)
            rmse_p5 = mean_squared_error(y_true_p5, y_pred_p5, squared=False)

            # 95th percentile
            p95 = np.percentile(y_true, 95.0)
            indx = [y_true >= p95][0]
            y_true_p95 = y_true[indx]
            y_pred_p95 = y_pred[indx]
            r2_p95 = r2_score(y_true_p95, y_pred_p95)
            rmse_p95 = mean_squared_error(
                y_true_p95, y_pred_p95, squared=False)

            # Print and append
            '''
            print('R2 = ', r2)
            print('RMSE = ', rmse)
            print('R2 p5 = ', r2_p5)
            print('RMSE p5 = ', rmse_p5)
            print('R2 p95 = ', r2_p95)
            print('RMSE p95 = ', rmse_p95)
            '''
            r2_list.append(r2)
            rmse_list.append(rmse)
            rmse_p5_list .append(rmse_p5)
            rmse_p95_list.append(rmse_p95)
            r2_p5_list .append(r2_p5)
            r2_p95_list .append(r2_p95)

    if ret is True:
        return [r2_list, rmse_list, r2_p5_list, rmse_p5_list, r2_p95_list, rmse_p95_list]


def single_location_comparison(location=[31.65, 77.34], station='Banjar',
                               min_year=2000, max_year=2010):
    """Plot model outputs for given coordinates over time."""

    aphro_ds = aphrodite.collect_APHRO(
        location, minyear=min_year, maxyear=max_year)
    cru_ds = cru.collect_CRU(location, minyear=min_year, maxyear=max_year)
    era5_ds = era5.collect_ERA5(location, minyear=min_year, maxyear=max_year)
    gpm_ds = gpm.collect_GPM(location,  minyear=min_year, maxyear=max_year)
    wrf_ds = beas_sutlej_wrf.collect_BC_WRF(
        location, minyear=min_year, maxyear=max_year)
    gauge_ds = beas_sutlej_gauges.gauge_download(
        station, minyear=min_year, maxyear=max_year)

    # cmip_ds = cmip5.collect_CMIP5()
    # cordex_ds = cordex.collect_CORDEX()
    # model_ts = model_prep([lat, lon], data_filepath='single_loc_test.csv', \
    # model_filepath=model_filepath)

    timeseries = [gauge_ds, gpm_ds, era5_ds, wrf_ds, aphro_ds, cru_ds]

    tims.benchmarking_subplots(timeseries, reference_dataset=gauge_ds)
    dataset_stats(timeseries, ref_ds=gauge_ds)
    # corr.dataset_correlation(timeseries)
    # pdf.benchmarking_plot(timeseries, kernel_density=False)


def basin_comparison(model_filepath, location):
    """ Plot model outputs for given basin over time."""

    aphro_ds = aphrodite.collect_APHRO(location, minyear=2000, maxyear=2010)
    cru_ds = cru.collect_CRU(location, minyear=2000, maxyear=2010)
    era5_ds = era5.collect_ERA5(location, minyear=2000, maxyear=2010)
    gpm_ds = gpm.collect_GPM(location,  minyear=2000, maxyear=2010)
    wrf_ds = beas_sutlej_wrf.collect_BC_WRF(
        location, minyear=2000, maxyear=2010)

    # cmip_ds = cmip5.collect_CMIP5()
    # cordex_ds = cordex.collect_CORDEX()
    # cmip_bs = select_basin(cmip_ds, location)
    # cordex_bs = select_basin(cordex_ds, location)
    # model_bs = model_prep(location, model_filepath)

    basins = [aphro_ds, cru_ds, era5_ds, gpm_ds, wrf_ds]

    dataset_stats(basins)


def multi_location_comparison():
    """Plot model outputs for multiple locations over time."""

    gauge_ds = beas_sutlej_gauges.all_gauge_data(
        minyear=2000, maxyear=2010, threshold=3653)
    locations = [[31.424, 76.417], [31.357, 76.878], [31.52, 77.22],
                 [31.67, 77.06], [31.454, 77.644], [31.238, 77.108],
                 [31.65, 77.34], [31.88, 77.15], [31.77, 77.31],
                 [31.80, 77.19]]

    aphro_sets = []
    cru_sets = []
    era5_sets = []
    gpm_sets = []
    wrf_sets = []

    for loc in locations:
        aphro_ds = aphrodite.collect_APHRO(loc, minyear=2000, maxyear=2010)
        aphro_sets.append(aphro_ds.tp)

        cru_ds = cru.collect_CRU(loc, minyear=2000, maxyear=2010)
        cru_sets.append(cru_ds.tp)

        era5_ds = era5.collect_ERA5(loc, minyear=2000, maxyear=2010)
        era5_sets.append(era5_ds.tp)

        gpm_ds = gpm.collect_GPM(loc,  minyear=2000, maxyear=2010)
        gpm_sets.append(gpm_ds.tp)

        wrf_ds = beas_sutlej_wrf.collect_BC_WRF(
            loc, minyear=2000, maxyear=2010)
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
    print(timeseries)
    pdf.mult_gauge_loc_plot(gauge_ds, timeseries)


def gauge_stats(minyear: str, maxyear: str) -> tuple:
    """
    Statisitics for gauges in the Uppper Beas and Sutlej Basins

    Print mean, standard deviations and slope for datasets.
    As well as calculating R2 and average RMSE, 5th percentile
    RMSE and 95th percentile RMSE for each reference dataset.
    These are in order:
     - ERA5
     - GPM
     - APHRODITE
     - CRU
     - Bias-corrected WRF

    Args:
        minyear(int): minimum year to analyse (inclusive)
        maxyear(int): maximum year to analyse (exclusive)

    Returns:
        tuple: list of averages and standerd deviations for each metric
    """

    bs_station_df = pd.read_csv(
        data_dir + '/bs_gauges/bs_only_gauge_info.csv')
    bs_station_df = bs_station_df.set_index('Unnamed: 0')
    station_list = list(bs_station_df.T)

    r2_list = []
    rmse_list = []
    rmse_p5_list = []
    rmse_p95_list = []
    r2_p5_list = []
    r2_p95_list = []

    for s in tqdm(station_list):

        gauge_ds = beas_sutlej_gauges.gauge_download(
            s, minyear=minyear, maxyear=maxyear)

        location = bs_station_df.loc[s].values
        # print(location)

        aphro_ds = aphrodite.collect_APHRO(location, minyear, maxyear)
        cru_ds = cru.collect_CRU(location, minyear, maxyear)
        era5_ds = era5.collect_ERA5(location, minyear, maxyear)
        gpm_ds = gpm.collect_GPM(location,  minyear, maxyear)
        wrf_ds = beas_sutlej_wrf.collect_BC_WRF(location, minyear, maxyear)

        timeseries = [era5_ds, gpm_ds, aphro_ds, cru_ds, wrf_ds]

        #print(era5_ds, gpm_ds, aphro_ds, cru_ds, wrf_ds)
        # Function to calculate statistics for each dataset and print values
        r2s, rmses, r2_p5, rmses_p5, r2_p95, rmses_p95 = dataset_stats(
            timeseries, ref_ds=gauge_ds, ret=True)
        r2_list.append(r2s)
        rmse_list.append(rmses)
        rmse_p5_list.append(rmses_p5)
        rmse_p95_list.append(rmses_p95)
        r2_p5_list.append(r2_p5)
        r2_p95_list.append(r2_p95)

    avg_r2 = np.array(r2_list).mean(axis=0)
    avg_rmse = np.array(rmse_list).mean(axis=0)
    avg_r2_p5 = np.array(r2_p5_list).mean(axis=0)
    avg_rmse_p5 = np.array(rmse_p5_list).mean(axis=0)
    avg_r2_p95 = np.array(r2_p95_list).mean(axis=0)
    avg_rmse_p95 = np.array(rmse_p95_list).mean(axis=0)

    std_r2 = np.array(r2_list).std(axis=0)
    std_rmse = np.array(rmse_list).std(axis=0)
    std_r2_p5 = np.array(r2_p5_list).std(axis=0)
    std_rmse_p5 = np.array(rmse_p5_list).std(axis=0)
    std_r2_p95 = np.array(r2_p95_list).std(axis=0)
    std_rmse_p95 = np.array(rmse_p95_list).std(axis=0)

    avgs = [avg_r2, avg_rmse, avg_r2_p5, avg_rmse_p5, avg_r2_p95, avg_rmse_p95]
    stds = [std_r2, std_rmse, std_r2_p5, std_rmse_p5, std_r2_p95, std_rmse_p95]

    return avgs, stds
