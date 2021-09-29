# Multi fidelity modeling

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import GPy

from load import era5
import data_prep as dp

import emukit
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel


def hf_lf_data_plot(lf_arr, hf_arr):

    x_max = lf_arr.max()
    x = np.linspace(0, x_max, 2)

    plt.figure(figsize=(10, 10))
    plt.scatter(lf_arr, hf_arr, c='purple', label='HF-LF Correlation')
    plt.plot(x, x, 'k--', alpha=0.25)
    plt.xlabel('LF(x)')
    plt.ylabel('HF(x)')
    plt.legend()
    plt.show()


def linear_model(X_train, Y_train, save=False):
    """ Returns GPy model of linear multi fidelity GP (Kennedy and O'Hagan 2000) """

    kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1)]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(
        kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(
        X_train, Y_train, lin_mf_kernel, n_fidelities=2)

    # Fix noise to kernel
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)

    # Optimise
    lin_mf_model = GPyMultiOutputWrapper(
        gpy_lin_mf_model, 2, n_optimization_restarts=5)
    lin_mf_model.optimize()

    if save is notFalse:
        lin_mf_model.save_model(save)

    return lin_mf_model


def nonlinear_model(X_train, Y_train, save=False):
    """ Returns GPy model of nonlinear multi-fidelity GP (Perdikaris 2017) """

    base_kernel = GPy.kern.RBF
    kernels = make_non_linear_kernels(base_kernel, 2, X_train.shape[1] - 1)
    nonlin_mf_model = NonLinearMultiFidelityModel(
        X_train, Y_train, n_fidelities=2, kernels=kernels, verbose=True, optimization_restarts=5)

    for m in nonlin_mf_model.models:
        m.Gaussian_noise.variance.fix(0)

    # Optimise
    nonlin_mf_model.optimize()

    if save is notFalse:
        nonlin_mf_model.save_model(save)

    return nonlin_mf_model


def unseen_data_plot(x_train_l, y_train_l, x_train_h, y_train_h, xval, yval, x_plot):

    plt.figure(figsize=(20, 20))

    plt.scatter(x_train_l[150:200], y_train_l[150:200], c='darkgreen')
    plt.scatter(x_train_h[150:200], y_train_h[150:200], c='darkorange')
    plt.scatter(x_val, y_val, c='r', label='Unseen High Fidelity Data')

    plt.plot(x_plot, lf_mean_model, '--g', label='Predicted Low Fidelity')
    plt.plot(x_plot, hf_mean_model, '--y', label='Predicted High Fidelity')

    oned_x_plot = x_plot.flatten()
    oned_lf_mean_mf_model = lf_mean_mf_model.flatten()
    oned_lf_std_mf_model = lf_std_mf_model.flatten()
    oned_hf_mean_mf_model = hf_mean_mf_model.flatten()
    oned_hf_std_mf_model = hf_std_mf_model.flatten()

    plt.fill_between(oned_x_plot, oned_lf_mean_mf_model - 1.96 * oned_lf_std_mf_model,
                     oned_lf_mean_mf_model + 1.96 * oned_lf_std_mf_model, color='mediumseagreen', alpha=0.2)
    plt.fill_between(oned_x_plot, oned_hf_mean_mf_model - 1.96 * oned_hf_std_mf_model,
                     oned_hf_mean_mf_model + 1.96 * oned_hf_std_mf_model, color='y', alpha=0.2)

    plt.xlabel('x')
    plt.ylabel('Precipitation (mm/day)')
    plt.legend()
    plt.show()


def metrics(y_val, y_pred):

    RMSE = mean_squared_error(y_val, y_pred)
    print(np.sqrt(RMSE))

    R2 = r2_score(y_val, y_pred)
    print(R2)
