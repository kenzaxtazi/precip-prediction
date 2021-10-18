# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import GPy

from load import beas_sutlej_gauges, era5
import gp.data_prep as dp
from sklearn.metrics import r2_score

import emukit
from emukit.multi_fidelity.convert_lists_to_array \
    import convert_xy_lists_to_arrays, convert_x_list_to_array
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel

# Lists

station_list = ['Banjar', 'Bhakra', 'Larji', 'Kasol',
                'Sainj', 'Suni', 'Pandoh', 'Janjehl', 'Bhuntar', 'Rampur']
altitude_list = [1914, 518, 975, 662, 1280, 655, 899, 2071, 1100, 976]
lat_list = [31.65, 31.424, 31.8, 31.357,
            31.77, 31.238, 31.67, 31.52, 31.88, 31.454]
lon_list = [77.34, 76.417, 77.19, 76.878,
            77.31, 77.108, 77.06, 77.22, 77.15, 77.644]
slope_list = []
anor_list = []

# Linear MFDGP


def linear_mfdgp(X_train, Y_train):
    kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1)]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(
        kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(
        X_train, Y_train, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
    gpy_lin_mf_model.optimize_restarts(num_restarts=5)
    return gpy_lin_mf_model


def test_log_likelihood(model, X_test, y_test):
    """ Marginal log likelihood for GPy model on test data"""
    _, test_log_likelihood, _ = model.inference_method.inference(
        model.kern.rbf_1, X_test, model.likelihood.Gaussian_noise_1, y_test,
        model.mean_function, model.Y_metadata)
    return test_log_likelihood


hf_heldout_ll = []
hf_train_ll = []
R2_hf_list = []
R2_lf_list = []

# Calculate values

for station in station_list:

    # Prepare data
    hf_ds = beas_sutlej_gauges.gauge_download(
        station, minyear=1980, maxyear=2011)
    lf_ds = era5.gauge_download(station, minyear=1980, maxyear=2011)

    hf_ds = hf_ds.assign_coords({"time": lf_ds.time[:len(hf_ds.time.values)]})

    hf_ds = hf_ds.dropna(dim='time')
    lf_ds = lf_ds.dropna(dim='time')

    # Transform data
    hf_ds['tp_tr'] = dp.log_transform(hf_ds['tp'].values)
    lf_ds['tp_tr'] = dp.log_transform(lf_ds['tp'].values)

    # High fidelity data needs to be smaller in length then low fidelity data
    x_train_l = lf_ds.time[:330].values.reshape(-1, 1)
    x_train_h = hf_ds.time[:240].values.reshape(-1, 1)

    y_train_l = lf_ds.tp[:330].values.reshape(-1, 1)
    y_train_h = hf_ds.tp[:240].values.reshape(-1, 1)

    x_val = hf_ds.time[240:330].values.reshape(-1, 1)
    y_val = hf_ds.tp[240:330].values.reshape(-1, 1)

    X_train, Y_train, = convert_xy_lists_to_arrays(
        [x_train_l, x_train_h], [y_train_l, y_train_h])

    # Train model
    lin_mf_model = linear_mfdgp(X_train, Y_train)

    # Training log likelihood
    train_log_lik = lin_mf_model.log_likelihood()
    hf_train_ll.append(train_log_lik)

    # Heldout log likelihood
    x_met = convert_x_list_to_array([x_val, x_val])
    n = x_val.shape[0]
    held_log_lik = test_log_likelihood(lin_mf_model, x_met[:n], y_val)
    hf_heldout_ll.append(held_log_lik)

    # R2
    wrp_model = GPyMultiOutputWrapper(
        lin_mf_model, 2, n_optimization_restarts=1)
    lin_mf_l_y_pred, mf_l_y_std_pred = wrp_model.predict(x_met[:n])
    lin_mf_h_y_pred, mf_h_y_std_pred = wrp_model.predict(x_met[n:])
    R2_hf = r2_score(y_val, lin_mf_h_y_pred)
    R2_lf = r2_score(y_val, lin_mf_l_y_pred)
    R2_hf_list.append(R2_hf)
    R2_lf_list.append(R2_lf)

plt.figure(figsize=(7, 7))
plt.scatter(R2_lf_list, R2_hf_list, s=50)
plt.scatter(0, 0, marker='+', color='r', s=100)
x = np.linspace(-2, 1, 2)
plt.plot(x, x, linestyle='--', color='grey')
plt.xlabel('Low-fidelity R2')
plt.ylabel('High-fidelity R2')

plt.show()
