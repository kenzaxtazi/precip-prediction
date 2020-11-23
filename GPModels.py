# GP models

import datetime
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

import DataExploration as de
import DataPreparation as dp
import FileDownloader as fd
import Metrics as me

import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary


# Filepaths and URLs

mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"
khyber_mask = "Data/Khyber_mask.nc"
gilgit_mask = "Data/Gilgit_mask.nc"
ngari_mask = "Data/Ngari_mask.nc"


"""
# Single point multivariate GP preparation
xtrain, xval, xtest, ytrain, yval, ytest = dp.multivariate_data_prep()

# Random sampling multivariate GP preparation
xtrain, xval, xtest, ytrain, yval, ytest = dp.random_multivariate_data_prep()
"""


def multi_gp(xtrain, xval, ytrain, yval, save=False):
    """ Returns model and plot of GP model using sklearn """

    # model construction
    k1 = gpflow.kernels.Periodic(
        gpflow.kernels.Matern52(lengthscales=0.3, variance=1, active_dims=[0])
    )
    k2 = gpflow.kernels.Matern52(active_dims=np.arange(1, len(xval[0])))
    # k3 = gpflow.kernels.White()

    k = k1 + k2  # +k

    # mean_function = gpflow.mean_functions.Linear(A=np.ones((len(xtrain[0]), 1)), b=[1])

    m = gpflow.models.GPR(
        data=(xtrain, ytrain.reshape(-1, 1)), kernel=k) #, mean_function=mean_function)

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        m.training_loss, m.trainable_variables) #, options=dict(maxiter=1000)
    # print_summary(m)

    x_plot = np.concatenate((xtrain, xval))
    y_gpr, y_std = m.predict_y(x_plot)

    print(
        " {0:.3f} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} |".format(
            me.R2(m, xtrain, ytrain),
            me.RMSE(m, xtrain, ytrain),
            me.R2(m, xval, yval),
            me.RMSE(m, xval, yval),
            np.mean(y_gpr),
            np.mean(y_std),
        )
    )

    if save == True:
        filepath = save_model(m, xval, "")
        print(filepath)

    return m


def save_model(model, xval, qualifiers=None):  # TODO
    """ Save the model for future use, returns filepath """

    now = datetime.datetime.now()
    samples_input = xval

    frozen_model = gpflow.utilities.freeze(model)

    module_to_save = tf.Module()
    predict_fn = tf.function(
        frozen_model.predict_f,
        input_signature=[
            tf.TensorSpec(shape=[None, np.shape(xval)[1]], dtype=tf.float64)
        ],
    )
    module_to_save.predict_y = predict_fn

    samples_input = tf.convert_to_tensor(samples_input, dtype="float64")
    # original_result = module_to_save.predict(samples_input)

    filename = "model_" + now.strftime("%Y-%m-%D-%H-%M-%S")  # + qualifiers

    save_dir = "Models/" + filename
    tf.saved_model.save(module_to_save, save_dir)

    return save_dir


def restore_model(model_filepath):
    """ Restore a saved model """
    loaded_model = tf.saved_model.load(model_filepath)
    return loaded_model
