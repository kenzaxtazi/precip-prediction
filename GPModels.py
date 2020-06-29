# GP models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DataExploration as de
import DataPreparation as dp 
import FileDownloader as fd
import Clustering as cl
import Metrics as me

import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary

# Filepaths and URLs
    
mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'
khyber_mask = 'Khyber_mask.nc'
gilgit_mask = 'Gilgit_mask.nc'
ngari_mask = 'Ngari_mask.nc'

tp_filepath = 'Data/era5_tp_monthly_1979-2019.nc'
tp_ensemble_filepath = 'Data/adaptor.mars.internal-1587987521.7367163-18801-5-5284e7a8-222a-441b-822f-56a2c16614c2.nc'

'''
# Single point GP preparation
da = dp.apply_mask(tp_ensemble_filepath, mask_filepath)
x_train, y_train, dy_train, x_test, y_test, dy_test = dp.point_data_prep()

# Single point multivariate GP preparation
xtrain, xval, xtest, ytrain, yval, ytest = dp.multivariate_data_prep()

# Area multivariate GP preparation
xtrain, xval, xtest, ytrain, yval, ytest = dp.area_data_prep(khyber_mask)
'''

## GPflow 

def gpflow_gp(x_train, y_train, dy_train, x_test, y_test, dy_test):
    """ Returns model and plot of GP model using GPflow """

    # model construction
    k1 = gpflow.kernels.Periodic(gpflow.kernels.RBF(lengthscales=0.3, variance=8, active_dims=[0]))
    k2 = gpflow.kernels.RBF(lengthscales=0.03, variance=1)
    k = k1 + k2

    mean_function = gpflow.mean_functions.Linear(A=None, b=None)

    #likelihood = gpflow.likelihoods.Gaussian(variance_lower_bound= 0.5)

    m = gpflow.models.GPR(data=(x_train, y_train.reshape(-1,1)), kernel=k1, mean_function=mean_function)
    # m.likelihood.variance.set_trainable(False)

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables)
    print_summary(m)

    x_predictions = np.linspace(0,40,2000)
    x_plot = x_predictions.reshape(-1, 1)
    y_gpr, y_std = m.predict_y(x_plot)


    ## generate 10 samples from posterior
    tf.random.set_seed(1)  # for reproducibility
    samples = m.predict_f_samples(x_plot, 10)  # shape (10, 100, 1)

    plt.figure()
    plt.title('GPflow fit for Gilgit (35.8884°N, 74.4584°E, 1500m)')
    plt.errorbar(x_train + 1981, y_train, dy_train, fmt='k.', capsize=2, label='ERA5 training data')
    plt.errorbar(x_test + 1981, y_test, dy_test, fmt='g.', capsize=2, label='ERA5 testing data')
    plt.plot(x_predictions + 1981, y_gpr[:, 0], color='orange', linestyle='-', label='Prediction')
    plt.fill_between(x_predictions + 1981, y_gpr[:, 0] - 1.9600 * y_std[:, 0], y_gpr[:, 0] + 1.9600 * y_std[:, 0],
             alpha=.2, label='95% confidence interval')
    #plt.plot(x_predictions + 1981, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    plt.legend()
    plt.ylabel('Precipitation [mm/day]')
    plt.xlabel('Year')
    plt.show()         

    return m 


def area_gpflow_gp(xtrain, ytrain, xval, yval, name=None):
    """ Returns model and plot of GP model using GPflow for a given area"""

    # model construction
    k1 = gpflow.kernels.Periodic(gpflow.kernels.RBF(lengthscales= 0.3,  variance=6, active_dims=[0]))
    k2 = gpflow.kernels.RBF()
    k = k1 + k2 

    mean_function = gpflow.mean_functions.Linear(A=np.ones((len(xtrain[0]),1)), b=[1])
    m = gpflow.models.GPR(data=(xtrain, ytrain.reshape(-1,1)), kernel=k2, mean_function=mean_function)

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

    x_plot = np.concatenate((xtrain, xval))
    y_plot = np.concatenate((ytrain, yval))
    y_gpr, y_std = m.predict_y(x_plot)

    ## generate 10 samples from posterior
    tf.random.set_seed(1)  # for reproducibility
    samples = m.predict_f_samples(x_plot, 10)  # shape (10, 100, 1)

    ## Take mean accross latitude and longitude

    x_df = pd.DataFrame(x_plot[:, 0:3], columns=['latitude', 'longitude', 'time'])
    y_df = pd.DataFrame(y_plot, columns=['tp true'])
    y_pred_df = pd.DataFrame(np.array(y_gpr), columns=['tp pred'])
    std_pred_df = pd.DataFrame(np.array(y_std), columns=['std pred'])

    results_df = pd.concat([x_df, y_df, y_pred_df, std_pred_df], axis=1)
    plot_df = results_df.groupby(['time']).mean()

    ## Figure 
    plt.figure()
    plt.title( name + 'Cluster fit')
    # plt.errorbar(x_train[:,0] + 1979, y_train, dy_train, fmt='k.', capsize=2, label='ERA5 training data')
    # plt.errorbar(x_test[:,0] + 1979, y_test, dy_test, fmt='g.', capsize=2, label='ERA5 testing data')
    plt.scatter(plot_df.index + 1979, plot_df['tp true'])
    plt.plot(plot_df.index + 1979, plot_df['tp pred'], color='orange', linestyle='-', label='Prediction')
    plt.fill_between(plot_df.index + 1979, plot_df['tp pred'] - 1.9600 * plot_df['std pred'], plot_df['tp pred'] + 1.9600 * plot_df['std pred'],
             alpha=.2, label='95% confidence interval')
    #plt.plot(x_predictions + 1981, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    plt.legend()
    plt.ylabel('Precipitation [mm/day]')
    plt.xlabel('Year')
    plt.show()

    return m


def multi_gpflow_gp(xtrain, xval, ytrain, yval):
    """ Returns model and plot of GP model using sklearn """

    # model construction
    k1 = gpflow.kernels.Periodic(gpflow.kernels.RBF(lengthscales=0.3,  variance=6, active_dims=[0]))
    k2 = gpflow.kernels.RBF()
    # k3 = gpflow.kernels.White()
  
    k = k1 + k2  #+k3

    mean_function = gpflow.mean_functions.Linear(A=np.ones((len(xtrain[0]),1)), b=[1])

    #likelihood = gpflow.likelihoods.Gaussian(variance_lower_bound= 0.5)

    m = gpflow.models.GPR(data=(xtrain, ytrain.reshape(-1,1)), kernel=k, mean_function=mean_function)
    # print_summary(m)
    # print(m.training_loss)
    # print(m.trainable_variables)
    # m.likelihood.variance.set_trainable(False)

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
    print_summary(m)

    x_plot = np.concatenate((xtrain, xval))
    y_gpr, y_std = m.predict_y(x_plot)

    ## generate 10 samples from posterior
    tf.random.set_seed(1)  # for reproducibility
    samples = m.predict_f_samples(x_plot, 10)  # shape (10, 100, 1)
    '''
    plt.figure()
    plt.title('GPflow fit for Gilgit (35.8884°N, 74.4584°E, 1500m)')
    #plt.errorbar(x_train[:,0] + 1979, y_train, dy_train, fmt='k.', capsize=2, label='ERA5 training data')
    #plt.errorbar(x_test[:,0] + 1979, y_test, dy_test, fmt='g.', capsize=2, label='ERA5 testing data')
    plt.plot(x_plot[:,0] + 1979, y_gpr, color='orange', linestyle='-', label='Prediction')
    plt.fill_between(x_plot[:,0]+ 1979, y_gpr[:, 0] - 1.9600 * y_std[:, 0], y_gpr[:, 0] + 1.9600 * y_std[:, 0],
             alpha=.2, label='95% confidence interval')
    plt.plot(x_plot[:,0] + 1979, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    plt.legend()
    plt.ylabel('Precipitation [mm/day]')
    plt.xlabel('Year')
    plt.show()
    '''
    
    print(" {0:.3f} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} |".format(me.R2(m, xtrain, ytrain), me.RMSE(m, xtrain, ytrain), me.R2(m, xval, yval), me.RMSE(m, xval, yval), np.mean(y_gpr), np.mean(y_std)))
    return m

