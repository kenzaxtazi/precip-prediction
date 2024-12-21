#!/usr/bin/env python
# coding: utf-8

# # Non-stationary model
# 5th February 2024

import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/precip-prediction/')
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/ckl/')
sys.path.append('/Users/kenzatazi/Documents/CDT/Code')

import tqdm
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs

import utils.metrics as me
import gp.data_prep as dp
from scipy.special import inv_boxcox, boxcox

from load import era5
import gpytorch
import torch

import matplotlib.pylab as plt
from sklearn.metrics import root_mean_squared_error, r2_score
from gp.gibbs_gp import MultiGibbsKernel

##############################


iteration = "_01"


##############


### Load data
dataset = dp.areal_model_new('uib', var="uib")
xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr = dataset.sets()

lmbda = dataset.l
yscaler = dataset.yscaler
xscaler = dataset.xscaler

### Transform coordinates to work better with log
xtrain_log = xtrain *20
xval_log = xval *20
xtest_log = xtest *20


### Model

# Training data
Xtrain, Ytrain_tr = torch.Tensor(xtrain_log).float(), torch.Tensor(ytrain_tr.reshape(-1)).float()

# Create inducing points for coordinate lengthscales
new_array = [tuple(row) for row in xtrain_log[:,1:3]]
z =  torch.Tensor(np.unique(new_array,axis=0))

# Model initialisation
likelihood = gpytorch.likelihoods.GaussianLikelihood() #noise=1e-3 * torch.ones(Xtrain.shape[0]))
likelihood.noise_covar.raw_noise.requires_grad = False
model = MultiGibbsKernel(Xtrain, Ytrain_tr, z, likelihood).double()

def MLL(y:np.ndarray, y_pred:np.ndarray, y_var:np.ndarray)-> float:
    """ Returns the mean log-loss score """
    MLL = np.mean(0.5 * np.log(2*np.pi*y_var) + 0.5 * (y - y_pred)**2 / y_var)
    return MLL




model.train()
likelihood.train()
    
n_iter = 600
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

losses = []

for i in tqdm.tqdm_notebook(range(n_iter)):
    with gpytorch.settings.max_cg_iterations(6000):
        optimizer.zero_grad()
        output = model(Xtrain)
        loss = -mll(output, Ytrain_tr)
        loss.backward()
        losses.append(loss.item())
        
        model.eval()
        likelihood.eval()
        if i%10 == 0:
            preds = likelihood(model(Xtrain))
            means = preds.loc.detach().numpy()
            r2 = r2_score(means, ytrain_tr[:2000])
            rmse = root_mean_squared_error(means, ytrain_tr[:2000])
            yvar = np.absolute(preds.covariance_matrix.diag().detach().numpy())
            mlloss = MLL(yval_tr, means, yvar)
            print(f'Iter {i + 1}/{n_iter} - Loss: {loss.item():.3f}, R2: {r2:.3f}, RMSE: {rmse:.3f}, MLL:{mlloss:.3f}, Noise: {model.likelihood.noise.item():.3f}, Var:{np.mean(yvar):.3f}')
        model.train()
        likelihood.train()
        
        optimizer.step()


### Lengthscales
H = model.base_covar_module.base_kernel.H
lengthscales = []

for l in range(394):
    l_loc = []
    for i in range(4):
        l_loc.append(H[i, l].detach().numpy())
    Xi = np.array(l_loc).reshape(2,2)
    lengthscales.append(np.sqrt(np.diagonal(Xi*Xi.T)))
lengthscales = np.array(lengthscales)
np.save(f'lengthscales{iteration}.npy', lengthscales)


### Predict

yval = np.nan_to_num(inv_boxcox(yscaler.inverse_transform(yval_tr[:].reshape(-1,1)), lmbda), nan=0)
ytrain = np.nan_to_num(inv_boxcox(yscaler.inverse_transform(ytrain_tr[:].reshape(-1,1)), lmbda), nan=0)
ytest = np.nan_to_num(inv_boxcox(yscaler.inverse_transform(ytest_tr[:].reshape(-1,1)), lmbda), nan=0)

model.eval()
likelihood.eval()

pred_yval = likelihood(model(torch.Tensor(xval_log).float()))
y_mean0_val = pred_yval.loc.detach()
y_var0_val = np.absolute(pred_yval.covariance_matrix.diag().detach())
y_mean_val = np.nan_to_num(inv_boxcox(yscaler.inverse_transform(y_mean0_val.reshape(-1,1)), lmbda), nan=0)

pred_ytrain = likelihood(model(Xtrain))
y_mean0_train = pred_ytrain.loc.detach()
y_var0_train = np.absolute(pred_ytrain.covariance_matrix.diag().detach())
y_mean_train = np.nan_to_num(inv_boxcox(yscaler.inverse_transform(y_mean0_train.reshape(-1,1)), lmbda), nan=0)

pred_ytest = likelihood(model(torch.Tensor(xtest_log).float()))
y_var0_test = np.absolute(pred_ytest.covariance_matrix.diag().detach())
y_mean0_test = pred_ytest.loc.detach()
y_mean_test = np.nan_to_num(inv_boxcox(yscaler.inverse_transform(y_mean0_test.reshape(-1,1)), lmbda), nan=0)

print('R2 train | RMSE train | MLL train | R2 val | RMSE val | MLL val |')

print(" {0:.3f} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} |".format(
        me.R2(ytrain, y_mean_train),
        me.RMSE(ytrain, y_mean_train),
        me.MLL(ytrain, y_mean_train, y_var0_train),
        me.R2(yval, y_mean_val),
        me.RMSE(yval, y_mean_val),
        me.MLL(yval, y_mean_val, y_var0_val)))
