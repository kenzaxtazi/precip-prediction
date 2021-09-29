# Model evaluation

import datetime
import numpy as np
import xarray as xr
import pandas as pd
import sklearn as sk
import cartopy.crs as ccrs
import matplotlib.cm as cm
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from tqdm import tqdm

import DataExploration as de
import DataPreparation as dp
import DataDownloader as dd
import Metrics as me
import GPModels as gpm
import Sampling as sa


mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"


def single_loc_evaluation(location, perf_plot=False, hpar_plot=False):

    metric_list = []
    coord_list = sa.random_location_generator(location)
    n = len(coord_list)

    for i in tqdm(range(n)):
        try:
            xtrain, xval, _, ytrain, yval, _ = dp.point_model(
                coords=list(coord_list[i])
            )
            m = gpm.multi_gp(xtrain, xval, ytrain, yval)

            training_R2 = me.R2(m, xtrain, ytrain)
            training_RMSE = me.RMSE(m, xtrain, ytrain)
            val_R2 = me.R2(m, xval, yval)
            val_RMSE = me.RMSE(m, xval, yval)
            time_kernel_lengthscale = float(
                m.kernel.kernels[0].base_kernel.variance.value())
            time_kernel_variance = float(
                m.kernel.kernels[0].base_kernel.lengthscales.value())
            time_kernel_periodicity = float(m.kernel.kernels[0].period.value())
            N34_lengthscale = np.array(
                m.kernel.kernels[1].lengthscales.value())[2]
            d2m_lengthscale = np.array(
                m.kernel.kernels[1].lengthscales.value())[0]
            tcwv_lengthscale = np.array(
                m.kernel.kernels[1].lengthscales.value())[1]
            rbf_kernel_variance = float(m.kernel.kernels[1].variance.value())

            metric_list.append(
                [
                    coord_list[i, 0],
                    coord_list[i, 1],
                    training_R2,
                    training_RMSE,
                    val_R2,
                    val_RMSE,
                    time_kernel_lengthscale,
                    time_kernel_variance,
                    time_kernel_periodicity,
                    N34_lengthscale,
                    d2m_lengthscale,
                    tcwv_lengthscale,
                    rbf_kernel_variance
                ]
            )

        except Exception:
            pass

    df = pd.DataFrame(
        metric_list,
        columns=[
            "latitude",
            "longitude",
            "training_R2",
            "training_RMSE",
            "val_R2",
            "val_RMSE",
            "time_kernel_lengthscale",
            "time_kernel_variance",
            "time_kernel_periodicity",
            "N34_lengthscale",
            "d2m_lengthscale",
            "tcwv_lengthscale",
            "rbf_kernel_variance"
        ],
    )

    now = datetime.datetime.now()
    df.to_csv("Data/single-locations-eval-" +
              now.strftime("%Y-%m-%d") + ".csv")

    print(df.mean(axis=0))

    df_prep = df.set_index(["latitude", "longitude"])
    da = df_prep.to_xarray()

    if perf_plot is True:
        slm_perf_plots(da)

    if hpar_plot is True:
        slm_hpar_plots(da)


def uib_evaluation(average=False):

    metric_list = []

    sample_list = [1000, 3000, 5000, 7000, 9000, 11000, 14000]

    for i in tqdm(sample_list):
        try:
            xtrain, xval, _, ytrain, yval, _ = dp.areal_model(
                length=i, EDA_average=average
            )
            m = gpm.multi_gp(xtrain, xval, ytrain, yval)

            training_R2 = me.R2(m, xtrain, ytrain)
            training_RMSE = me.RMSE(m, xtrain, ytrain)
            val_R2 = me.R2(m, xval, yval)
            val_RMSE = me.RMSE(m, xval, yval)

            metric_list.append(
                [i, training_R2, training_RMSE, val_R2, val_RMSE])

        except Exception:
            print(i + 100)
            xtrain, xval, _, ytrain, yval, _ = dp.areal_model(
                length=i + 100, EDA_average=average
            )
            m = gpm.multi_gp(xtrain, xval, ytrain, yval)

            training_R2 = me.R2(m, xtrain, ytrain)
            training_RMSE = me.RMSE(m, xtrain, ytrain)
            val_R2 = me.R2(m, xval, yval)
            val_RMSE = me.RMSE(m, xval, yval)

            metric_list.append(
                [i, training_R2, training_RMSE, val_R2, val_RMSE])

    df = pd.DataFrame(
        metric_list,
        columns=["samples", "training_R2",
                 "training_RMSE", "val_R2", "val_RMSE"],
    )
    df.to_csv("uib-eval-2020-07-22.csv")


def cluster_evaluation(mask):

    metric_list = []

    name = mask[0:6]

    sample_list = [1000, 3000, 5000, 7000, 9000, 11000, 14000]

    for i in tqdm(sample_list):
        try:
            xtrain, xval, _, ytrain, yval, _ = dp.areal_model(
                length=i, mask=mask
            )
            m = gpm.multi_gp(xtrain, xval, ytrain, yval)

            training_R2 = me.R2(m, xtrain, ytrain)
            training_RMSE = me.RMSE(m, xtrain, ytrain)
            val_R2 = me.R2(m, xval, yval)
            val_RMSE = me.RMSE(m, xval, yval)

            metric_list.append(
                [i, training_R2, training_RMSE, val_R2, val_RMSE])

        except Exception:
            print(i + 100)
            xtrain, xval, _, ytrain, yval, _ = dp.areal_model(
                length=i + 100, mask=mask
            )
            m = gpm.multi_gp(xtrain, xval, ytrain, yval)

            training_R2 = me.R2(m, xtrain, ytrain)
            training_RMSE = me.RMSE(m, xtrain, ytrain)
            val_R2 = me.R2(m, xval, yval)
            val_RMSE = me.RMSE(m, xval, yval)

            metric_list.append(
                [i, training_R2, training_RMSE, val_R2, val_RMSE])

    df = pd.DataFrame(
        metric_list,
        columns=["samples", "training_R2",
                 "training_RMSE", "val_R2", "val_RMSE"],
    )
    df.to_csv(name + "-eval-2020-07-22.csv")


def sampled_points(
    filepath1, label1, filepath2=None, label2=None, filepath3=None, label3=None
):
    """ Returns plots of R2 and RMSE as function of sampled data points """

    df1 = pd.read_csv(filepath1)

    if filepath2 is notNone:
        df2 = pd.read_csv(filepath2)

    if filepath3 is notNone:
        df3 = pd.read_csv(filepath3)

    plt.figure("R2")
    plt.scatter(df1["samples"].values,
                df1["R2_val"].values, c="b", label=label1)
    plt.plot(df1["samples"].values,
             df1["R2_val"].values, c="b", linestyle="--")
    if filepath2 is notNone:
        plt.scatter(df2["samples"].values,
                    df2["R2_val"].values, c="r", label=label2)
        plt.plot(df2["samples"].values,
                 df2["R2_val"].values, c="r", linestyle="--")
    if filepath3 is notNone:
        plt.scatter(df3["samples"].values,
                    df3["R2_val"].values, c="g", label=label3)
        plt.plot(df3["samples"].values,
                 df3["R2_val"].values, c="g", linestyle="--")
    plt.xlabel("Number of samples")
    plt.ylabel("R2")
    plt.legend()

    plt.figure("RMSE")
    plt.scatter(df1["samples"].values,
                df1["RMSE_val"].values, c="b", label=label1)
    plt.plot(df1["samples"].values,
             df1["RMSE_val"].values, c="b", linestyle="--")
    if filepath2 is notNone:
        plt.scatter(df2["samples"].values,
                    df2["RMSE_val"].values, c="r", label=label2)
        plt.plot(df2["samples"].values,
                 df2["RMSE_val"].values, c="r", linestyle="--")
    if filepath3 is notNone:
        plt.scatter(df3["samples"].values,
                    df3["RMSE_val"].values, c="g", label=label3)
        plt.plot(df3["samples"].values,
                 df3["RMSE_val"].values, c="g", linestyle="--")
    plt.xlabel("Number of samples")
    plt.ylabel("RMSE [mm/day]")
    plt.legend()

    plt.show()


def slm_perf_plots(da):
    """ Plots figures for Single Location Model """

    plt.figure("SLM R2")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    da.val_R2.plot(
        center=0.0,
        vmax=1.0,
        cbar_kwargs={"label": "\n Validation R2",
                     "extend": "neither", "pad": 0.10},
    )
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.figure("SLM RMSE")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    da.val_RMSE.plot(
        cbar_kwargs={"label": "\n Validation RMSE",
                     "extend": "neither", "pad": 0.10}
    )
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.show()


def slm_hpar_plots(da):
    """
    plt.figure("SLM time kernel lengthscale")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    da.time_kernel_lengthscale.plot(robust=True,
        cbar_kwargs={"label": "\n Time axis kernel lengthscale", "extend": "max", "pad": 0.10},
    )
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.figure("SLM time kernel variance")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    da.time_kernel_variance.plot(robust=True, 
        cbar_kwargs={"label": "\n Time axis kernel variance", "extend": "max", "pad": 0.10},
    )
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    """
    plt.figure("SLM time kernel periodicity")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    a = np.log10(da.time_kernel_periodicity)
    a.plot(robust=True, vmin=0,
           cbar_kwargs={
               "label": "\n Time axis kernel periodicity (10^x)", "extend": "neither", "pad": 0.10},
           )
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    """
    plt.figure("SLM RBF kernel variance")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    da.rbf_kernel_variance.plot(robust=True, 
        cbar_kwargs={"label": "\n RBF kernel variance", "extend": "max", "pad": 0.10},
    )
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    plt.figure("N34 lengthscale")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    a = np.log10(da.N34_lengthscale)
    a.plot(robust=True, vmin=0,
        cbar_kwargs={"label": "\n N34 index lengthscale (10^x)", "extend": "neither", "pad": 0.10},
    )
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    plt.figure("tcwv lengthscale")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    da.tcwv_lengthscale.plot(robust=True,
        cbar_kwargs={"label": "\n tcwv lengthscale", "extend": "max", "pad": 0.10},
    )
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.figure("d2m lengthscale")
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    da.d2m_lengthscale.plot(robust=True,
        cbar_kwargs={"label": "\n d2m lengthscale", "extend": "max", "pad": 0.10},
    )
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    """
    plt.show()


def test_log_likelihood(model, X_test, y_test):
    """ Marginal log likelihood for GPy model on test data"""
    _, test_log_likelihood, _ = model.inference_method.inference(
        model.kern, X_test, model.likelihood, y_test, model.mean_function, model.Y_metadata)
    return test_log_likelihood
