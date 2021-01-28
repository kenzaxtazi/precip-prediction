# Timeseries

"""
A collection of functions to analyse timeseries. 
Assumes timeseries are pre-processed, i.e. they are DataArray objects with precipiation in mm/day and time in years.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import xarray as xr

import FileDownloader as fd
import DataDownloader as dd

from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def lin_reg(timeseries):
    """ 
    Outputs parameter of linear model
    
    Inputs
        timeseries: data array

    Outputs
        lineat_model: list [slope, intercept, r_value, p_value, std_err]
    """
    linear_model= stats.linregress(timeseries['time'].values, timeseries.values)

    print('Linear model parameters for '+ str(timeseries.lat.values) + "°N, " + str(timeseries.lon.values) + "°E")
    print('Slope', linear_model[0])
    print('Intercept', linear_model[1])
    print('R value', linear_model[2])
    print('P value', linear_model[3])
    print('Standard error', linear_model[4])

    return linear_model


def linreg_plot(timeseries, linear_models):
    """ 
    Returns plot of linear regression on one ore more timeseries
    """

    N = len(timeseries)
    _, axs = plt.subplots(N, sharex=True, sharey=True)

    for n in range(N):

        (slope, intercept, _, p_value, std_err) = linear_models[n]  

        time = timeseries[n].time.values
        axs[n].plot(timeseries[n].time.values, timeseries[n].values)
        axs[n].set_title( str(timeseries[n].lat.values) + "°N, " + str(timeseries[n].lon.values) + "°E")
        axs[n].plot(
            time,
            slope * time + intercept,
            color="green",
            linestyle="--",
            label="Slope = %.3f±%.3f mm/day/year, p-value = %.3f"
            % (slope, std_err, p_value),
        )
        axs[n].set_xlabel(" ")
        axs[n].set_ylabel("Total precipation [mm/day]")
        axs[n].grid(True)
        axs[n].legend()
    
    axs[n].set_xlabel("Year")
    plt.show()


def uib_sample_linreg():
    """ Plots sample timeseries for UIB clusters """

    # Open data
    mask_filepath = "Data/Masks/ERA5_Upper_Indus_mask.nc"
    tp = dd.download_data(mask_filepath, xarray=True)
    tp_da = tp.tp * 1000  # convert from m/day to mm/day

    ## Data
    gilgit = tp_da.interp(coords={"lon": 75, "lat": 36}, method="nearest")
    ngari = tp_da.interp(coords={"lon": 81, "lat": 32}, method="nearest")
    khyber = tp_da.interp(coords={"lon": 73, "lat": 34.5}, method="nearest")
    timeseries = [gilgit, ngari, khyber]

    gilgit_linear_model = lin_reg(gilgit)
    ngari_linear_model = lin_reg(ngari)
    khyber_linear_model = lin_reg(khyber)
    linear_models = [gilgit_linear_model, ngari_linear_model, khyber_linear_model]

    linreg_plot(timeseries, linear_models)


def benchmarking_plot(timeseries, xtr, y_gpr_t, y_std_t):
    """ 
    Plot timeseries of model outputs.
    Assumes that timeseries and model outputs are already formatted.
    """
    plt.figure()
    for ts in timeseries:
        plt.plot(ts.time.values, ts.tp.values, label=ts.plot_legend)
    plt.fill_between(xtr[:, 0] + 1970, y_gpr_t[:, 0] - 1.9600 * y_std_t[:, 0], y_gpr_t[:, 0] + 1.9600 * y_std_t[:, 0], 
                        alpha=0.5, color="lightblue") #label="95% confidence interval")
    plt.plot(xtr[:, 0] + 1970, y_gpr_t, linestyle="-", label="Prediction")
    plt.xlabel('Time')
    plt.ylabel('Precipitation mm/day')
    plt.legend()
    plt.show()



def rolling_timseries_comparison(timeseries, xtr, y_gpr_t, y_std_t):
    """ Plot rolling avereaged timeseries of model outputs """
    
    return 1


