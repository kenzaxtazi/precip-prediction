
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import PrecipitationDataExploration as pde
import FileDownloader as fd
import Clustering as cl

import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import torch
import gpytorch
import os

# Filepaths and URLs
    
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'

nao_url = 'https://www.psl.noaa.gov/data/correlation/nao.data'
n34_url =  'https://psl.noaa.gov/data/correlation/nina34.data'
n4_url =  'https://psl.noaa.gov/data/correlation/nina4.data'

tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
tp_ensemble_filepath ='/Users/kenzatazi/Downloads/adaptor.mars.internal-1587987521.7367163-18801-5-5284e7a8-222a-441b-822f-56a2c16614c2.nc'
mpl_filepath = '/Users/kenzatazi/Downloads/era5_msl_monthly_1979-2019.nc'


da = pde.apply_mask(tp_ensemble_filepath, mask_filepath)

def data_preparation(da):
    """ Outputs test and training data from DataArray """
    
    std_da = da.std(dim='number')
    mean_da = da.mean(dim='number')

    gilgit_mean = mean_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    gilgit_std = std_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')

    multi_index_df_mean = gilgit_mean.to_dataframe()
    df_mean= multi_index_df_mean.reset_index()
    df_mean_clean = df_mean.dropna()
    df_mean_clean['time'] = df_mean_clean['time'].astype('int')

    multi_index_df_std = gilgit_std.to_dataframe()
    df_std = multi_index_df_std.reset_index()
    df_std_clean = df_std.dropna()
    df_std_clean['time'] = df_std_clean['time'].astype('int')

    y = df_mean_clean['tp'].values*1000
    dy = df_std_clean['tp'].values*1000

    x_prime = df_mean_clean['time'].values.reshape(-1, 1)
    x = (x_prime - x_prime[0])/ (1e9*60*60*24*365)
    
    x_train = x[0:400]
    y_train = y[0:400]
    dy_train = dy[0:400]

    x_test = x[400:-2]
    y_test = y[400:-2]
    dy_test = dy[400:-2]

    return x_train, y_train, dy_train, x_test, y_test, dy_test


x_train, y_train, dy_train, x_test, y_test, dy_test = data_preparation(da)


def multi_data_prep():
    
    # Front indices
    nao_df = fd.update_url_data(nao_url, 'NAO')
    n34_df = fd.update_url_data(n34_url, 'N34')
    n4_df = fd.update_url_data(n4_url, 'N4')
    ind_df = nao_df.join([n34_df, n4_df]).astype('float64')


    # Orography, humidity and precipitation
    cds_filepath = fd.update_cds_data(variables=['2m_dewpoint_temperature', 'angle_of_sub_gridscale_orography', 
                                                'orography', 'slope_of_sub_gridscale_orography', 
                                                'total_column_water_vapour', 'total_precipitation'])
    masked_da = pde.apply_mask(cds_filepath, mask_filepath)
    gilgit = masked_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    multiindex_df = gilgit.to_dataframe()
    cds_df = multiindex_df.reset_index()

    # Combine
    df_combined = pd.merge_ordered(cds_df, ind_df, on='time')  
    df = df_combined.drop(columns=['expver', 'longitude', 'latitude'])
    df['time'] = df['time'].astype('int')
    df_clean = df.dropna()

    # Seperate y and x
    y = df_clean['tp'].values*1000

    x1 = df_clean.drop(columns=['tp'])
    x1['time'] = (x1['time'] - x1['time'].min())/ (1e9*60*60*24*365)
    x = x1.values.reshape(-1, 9)
    
    x_train = x[0:400]
    y_train = y[0:400]

    x_test = x[400:-2]
    y_test = y[400:-2]

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = multi_data_prep()


def sklearn_gp(x_train, y_train, dy_train):
    """ Returns model and plot of GP model using sklearn """

    kernel = ExpSineSquared(length_scale=1, periodicity=1)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=dy_train**2, random_state=0).fit(x_train, y_train)
    #print('R2 score = ', gpr.score(x_test, y_test))
  
    x_predictions = np.linspace(0,41,1000)
    x_plot = x_predictions.reshape(-1, 1)
    y_gpr, y_std = gpr.predict(x_plot, return_std=True)
    
    plt.figure()
    plt.title('GP fit for Gilgit (35.8884°N, 74.4584°E, 1500m)')
    plt.errorbar(x_train + 1981, y_train, dy_train, fmt='r.', markersize=10, label='ERA5')
    plt.plot(x_predictions + 1981, y_gpr, 'b-', label='Prediction')
    plt.fill_between(x_predictions + 1981, y_gpr - 1.9600 * y_std, y_gpr + 1.9600 * y_std,
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.legend()
    plt.ylabel('Precipitation [mm/day]')
    plt.xlabel('Year')
    plt.show()

    return gpr


## GPflow 

def multi_gpflow_gp(x_train, y_train, dy_train, x_test, y_test, dy_test):
    """ Returns model and plot of GP model using sklearn """

    # model construction
    k1 = gpflow.kernels.Periodic(gpflow.kernels.RBF(lengthscales= 0.3,  variance=6, active_dims=[0]))
    k2 = gpflow.kernels.RBF()
  
    k = k1 + k2 

    mean_function = gpflow.mean_functions.Linear(A=np.ones((9,1)), b=[1])

    #likelihood = gpflow.likelihoods.Gaussian(variance_lower_bound= 0.5)

    m = gpflow.models.GPR(data=(x_train, y_train.reshape(-1,1)), kernel=k2, mean_function=mean_function)
    # print_summary(m)
    # print(m.training_loss)
    # print(m.trainable_variables)
    # m.likelihood.variance.set_trainable(False)

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
    print_summary(m)

    x_plot = np.concatenate((x_train, x_test))
    y_gpr, y_std = m.predict_y(x_plot)

    ## generate 10 samples from posterior
    tf.random.set_seed(1)  # for reproducibility
    samples = m.predict_f_samples(x_test, 10)  # shape (10, 100, 1)

    plt.figure()
    plt.title('GPflow fit for Gilgit (35.8884°N, 74.4584°E, 1500m)')
    plt.errorbar(x_train[:,0] + 1981, y_train, dy_train, fmt='k.', capsize=2, label='ERA5 training data')
    plt.errorbar(x_test[:,0] + 1981, y_test, dy_test, fmt='g.', capsize=2, label='ERA5 testing data')
    plt.plot(x_plot[:,0] + 1981, y_gpr, color='orange', linestyle='-', label='Prediction')
    plt.fill_between(x_plot[:,0]+ 1981, y_gpr[:, 0] - 1.9600 * y_std[:, 0], y_gpr[:, 0] + 1.9600 * y_std[:, 0],
             alpha=.2, label='95% confidence interval')
    #plt.plot(x_predictions + 1981, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    plt.legend()
    plt.ylabel('Precipitation [mm/day]')
    plt.xlabel('Year')
    plt.show()

    return m


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


## GPyTorch

def gpytorch_gp(xtrain, ytrain, xtest, ytest):
    
    smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else 50

    x_train = torch.from_numpy(xtrain.astype(float))
    y_train = torch.from_numpy(ytrain.astype(float))

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.GammaPrior(.001, 100))
    model = ExactGPModel(x_train, y_train, likelihood)
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()}], # Includes GaussianLikelihood parameters
                                 lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()))
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        x_plot = torch.linspace(0, 40, 5000).type(torch.DoubleTensor)
        observed_pred = likelihood(model(x_plot))

    #### Plot ####

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(10, 3))
        plt.title('GP fit for Gilgit (35.8884°N, 74.4584°E, 1500m)')

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        
        # Plot training data as black stars
        ax.scatter(xtrain + 1979, ytrain, color='darkblue', marker='+', label='ERA5 training data')
        ax.scatter(xtest + 1979, ytest, color='maroon', marker='+', label='ERA5 test data')
        
        # Plot predictive means as blue line
        ax.plot(x_plot + 1979, observed_pred.mean.numpy(), 'b', label='Prediction')
        #ax.plot(xtest + 1979, observed_pred.mean.numpy(), 'b')

        # Shade between the lower and upper confidence bounds
        # ax.fill_between(xtrain.flatten() + 1979, lower.numpy(), upper.numpy(), alpha=0.2)
        ax.fill_between(x_plot.flatten() + 1979, lower.numpy(), upper.numpy(), alpha=0.2, label='2$\sigma$ confidence')
        ax.legend()

        plt.ylabel('Precipitation [mm/day]')
        plt.xlabel('Year')
        plt.show()

        return model


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel
                (period_length_prior=gpytorch.priors.NormalPrior(1, .001),
                lengthscale_prior=gpytorch.priors.NormalPrior(0.05, .001)),
                outputscale_prior= gpytorch.priors.NormalPrior(20, 20))
        
        # gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.NormalPrior(0.05, .001)))

        '''
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel
                (period_length_prior=gpytorch.priors.NormalPrior(1, .001),
                lengthscale_prior=gpytorch.priors.NormalPrior(0.05, .001)),
                outputscale_prior= gpytorch.priors.NormalPrior(2, 5))
        '''

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

