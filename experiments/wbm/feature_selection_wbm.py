#!/usr/bin/env python
# coding: utf-8

# # Whole basin model kernel design
# 16th April 2024



import gpflow
import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/precip-prediction/')

import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import tensorflow_probability as tfp
import tensorflow as tf

import utils.metrics as me
import gp.data_prep as dp
import gp.gp_models as gpm
from scipy.special import inv_boxcox

from load import era5, data_dir
import pickle



# ## Load


def new_kernel_list(old_kernel, iter=int):
    """ Build new kernels """

    new_kernel = gpflow.kernels.Matern32(lengthscales=1, variance=1, active_dims=[iter+3])

    k1 = old_kernel + new_kernel
    #k2 = old_kernel * new_kernel
    k3 = old_kernel

    kernel_list = [k1, k3] #k2
    
    return kernel_list

def fit_kernel(xtrain, xval, ytrain_tr, yval_tr, lmbda, kernel: gpflow.kernels.Kernel)-> tuple:
    """ Train model and return BIC and MLL for each location"""
        
    # train model
    m = gpm.multi_gp(xtrain, xval, ytrain_tr, yval_tr, lmbda_bc=lmbda, kernel=kernel, print_perf=False)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables)

    # Calculate Bayesian Information Criterion and Mean Log Likelihood
    bic = me.BIC(m, xval, yval_tr.reshape(-1, 1))
    return bic


xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr, lmbda= dp.areal_model_new(location='uib', length=2000, maxyear='2020')

# Locally periodic kernel wrt to time as we want period to be flexible 

k1 = gpflow.kernels.Periodic(gpflow.kernels.Matern32(lengthscales=1, variance=1, active_dims=[0]))
k1a = gpflow.kernels.Matern32(lengthscales=1, variance=1, active_dims=[0])
k2 = gpflow.kernels.Matern32(active_dims=[1,2])
ka = k1 * k1a + k2

# Lists
BIC_lists = []
kernel_list = [ka]

for j in tqdm.tqdm(range(16)):
    
    model_BIC_list = []

    for k in kernel_list:
        bic = fit_kernel(xtrain, xval, ytrain_tr, yval_tr, lmbda, k)
        model_BIC_list.append(bic)

    best_index = np.argmin(model_BIC_list)
    best_kernel = kernel_list[best_index]
    kernel_list = new_kernel_list(best_kernel, iter=j)
    BIC_lists.append(model_BIC_list)

with open("wbm_bic_list", "wb") as fp:   #Pickling
    pickle.dump(BIC_lists, fp)


BIC_arr = np.array(BIC_lists[1:])
x = np.arange(1, 16)


var_list = ["time, lat, lon", "tcwv", "slor", "d2m", "z", "EOF200U",  "t2m", 
            "EOF850U",  "EOF500U", "EOF500B2", "EOF200B", "anor", "NAO", 
            "EOF500U2", "N34", "EOF850U2", "EOF200B"]

plt.figure(figsize=(7,5))
plt.scatter(0, BIC_lists[0], marker='o', c='k')
plt.scatter(x, BIC_arr[:,0], marker='o', label='additive kernel')
plt.scatter(x, BIC_arr[:,1], marker='<', label='best previous kernel')
plt.ylabel('Bayesian Information Criterion')
plt.xlabel('Kernel design iterations')
plt.legend()
plt.xticks(np.arange(0, 16, 1), labels=var_list)
plt.savefig('wbm_bic.png')

