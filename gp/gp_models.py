# GP models

import datetime
import numpy as np

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import positive, print_summary
from scipy.special import inv_boxcox

import utils.metrics as me
import gp.data_prep as dp

# Filepaths and URLs
mask_filepath = "_Data/ERA5_Upper_Indus_mask.nc"
khyber_mask = "_Data/Khyber_mask.nc"
gilgit_mask = "_Data/Gilgit_mask.nc"
ngari_mask = "_Data/Ngari_mask.nc"


"""
# Single point multivariate GP preparation
xtrain, xval, xtest, ytrain, yval, ytest, lmbda = dp.point_model('uib')

# Random sampling multivariate GP preparation
xtrain, xval, xtest, ytrain, yval, ytest = dp.areal_model('uib', length=14000,
        maxyear=1993)
"""


def multi_gp(xtrain, xval, ytrain, yval, lmbda_bc, yscaler, kernel=None, save=False, print_perf=False):
    """ Returns simple GP model """
    
    if kernel == "point":
        # model construction
        k1 = gpflow.kernels.Periodic(gpflow.kernels.Matern32(
            lengthscales=1e-2, variance=1, active_dims=[0]), period=1./35.)
        k1b = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0])
        k = k1 * k1b 

        for i in np.arange(1, len(xval[0]+1)):
            k2 = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[i])
            k += k2
    
    if kernel == "areal":
        # model construction
        k1 = gpflow.kernels.Periodic(gpflow.kernels.Matern32(
            lengthscales=1e-2, variance=1, active_dims=[0]), period=1./35.)
        k1b = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0])
        ka = gpflow.kernels.Matern32(lengthscales=[1e-2,1e-2], variance=1, active_dims=[1,2])
        k = k1 * k1b + ka

        for i in np.arange(3, len(xval[0]+3)):
            k2 = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[i])
            k += k2

    #else:
    #k = kernel

    m = gpflow.models.GPR(data=(xtrain, ytrain.reshape(-1, 1)), kernel=k)
    #gpflow.set_trainable(m.kernel.kernels[0].kernels[0].kernels[0].period, False)
    #m.kernel.kernels[0].kernels[0].kernels[0].period.prior = tfp.distributions.Normal(loc=gpflow.utilities.to_default_float(2./35.), scale=gpflow.utilities.to_default_float(1e-2))
    #m.kernel.kernels[0].kernels[0].kernels[0].base_kernel.lengthscales.prior = tfp.distributions.Normal(loc=gpflow.utilities.to_default_float(1e-2), scale=gpflow.utilities.to_default_float(1e-2))

    opt = gpflow.optimizers.Scipy()
    # , options=dict(maxiter=1000)
    opt.minimize(m.training_loss, m.trainable_variables)
    print_summary(m)

    if print_perf is True:
        # Inverse transforms
        ytrain_inv_tr = inv_boxcox(yscaler.inverse_transform(ytrain), lmbda_bc)
        yval_inv_tr = inv_boxcox(yscaler.inverse_transform(yval), lmbda_bc)

        y_gpr_train0, y_var_train0 = m.predict_y(xtrain)
        y_gpr_train = inv_boxcox(yscaler.inverse_transform(y_gpr_train0), lmbda_bc)

        y_gpr_val0, y_var_val0 = m.predict_y(xval)
        y_gpr_val = inv_boxcox(yscaler.inverse_transform(y_gpr_val0), lmbda_bc)

        x_plot = np.concatenate((xtrain, xval))
        y_gpr_plot0, y_var_val0 = m.predict_y(x_plot)
        y_gpr_plot = inv_boxcox(yscaler.inverse_transform(y_gpr_plot0), lmbda_bc)

        print('R2 train | RMSE train | R2 val | RMSE val | mean | std |')

        print(
            " {0:.3f} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} |".format(
                me.R2(ytrain_inv_tr, y_gpr_train),
                me.RMSE(ytrain_inv_tr, y_gpr_train),
                me.R2(yval_inv_tr, y_gpr_val),
                me.RMSE(yval_inv_tr, y_gpr_val),
                np.mean(y_gpr_plot),
                np.std(y_gpr_plot),
            )
        )

    if save is not False:
        filepath = save_model(m, xval, save)
        print(filepath)

    return m


def hybrid_gp(xtrain, xval, ytrain, yval, save=False):
    """ Returns whole basin or cluster GP model with hybrid kernel """

    dimensions = len(xtrain[0])

    k1 = gpflow.kernels.RBF(lengthscales=np.ones(
        dimensions), active_dims=np.arange(0, dimensions))
    k2 = gpflow.kernels.RBF(lengthscales=np.ones(
        dimensions), active_dims=np.arange(0, dimensions))

    alpha1 = hybrid_kernel(dimensions, 1)
    alpha2 = hybrid_kernel(dimensions, 2)

    k = alpha1*k1 + alpha2*k2

    m = gpflow.models.GPR(data=(xtrain, ytrain.reshape(-1, 1)), kernel=k)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables)
    # print_summary(m)
    # TO DO inverse transform

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

    if save is True:
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

    filename = "model_" + now.strftime("%Y-%m-%D-%H-%M-%S") + qualifiers

    save_dir = "Models/" + filename
    tf.saved_model.save(module_to_save, save_dir)

    return save_dir


def restore_model(model_filepath):
    """ Restore a saved model """
    loaded_model = tf.saved_model.load(model_filepath)
    return loaded_model


class hybrid_kernel(gpflow.kernels.AnisotropicStationary):
    def __init__(self, dimensions, feature):
        super().__init__(active_dims=np.arange(dimensions))
        self.variance = gpflow.Parameter(1, transform=positive())
        self.feature = feature

    def K(self, X, X2):
        if X2 is None:
            X2 = X

        print('X ', np.shape(X))
        k = self.variance * (X[:, self.feature]
                             - X2[:, self.feature]) * tf.transpose(
            X[:, self.feature] - X2[:, self.feature])  # returns a 2D tensor
        print('k ', np.shape(k))
        return k

    def K_diag(self, X):
        print('X ', np.shape(X))
        kdiag = self.variance * (X[:, self.feature]) * \
            tf.transpose(X[:, self.feature])

        print('kdiag ', np.shape(kdiag))
        return kdiag  # this returns a 1D tensor


if __name__ in "__main__":
    # Single point multivariate GP preparation
    xtrain, xval, xtest, ytrain, yval, ytest = dp.point_model('uib')

    # Random sampling multivariate GP preparation
    xtrain, xval, xtest, ytrain, yval, ytest = dp.areal_model('uib', length=14000,
                                                              maxyear=1993)
