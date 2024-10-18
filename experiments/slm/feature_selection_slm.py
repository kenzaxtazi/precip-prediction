#!/usr/bin/env python
# coding: utf-8

# # Building kernel
# 16th April 2024

import gpflow
import sys
sys.path.append('/data/hpcdata/users/kenzi22/')
sys.path.append('/data/hpcdata/users/kenzi22/precip-prediction')
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf

import utils.metrics as me
import gp.data_prep as dp
from load import era5
import pickle

# ## Load 
data = era5.collect_ERA5('uib', minyear='1970', maxyear='1980', all_var=True)

df = data.to_dataframe()
loc_df = df.groupby(['lat','lon']).mean().reset_index()
loc_df.dropna(inplace=True)
locs = loc_df[['lon','lat']].values

tf.random.set_seed(42)

### Load

def new_kernel_list(old_dim_list, itera=int):
    """ Build new kernels """
    new_dim_list = old_dim_list + [itera+3]

    ### Old kernel
    k1a = gpflow.kernels.Periodic(gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0]), period=1./35.)
    k1b = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0])
    old_kernel = k1a * k1b

    if not old_dim_list:
        pass
    else:
        for i in np.arange(len(old_dim_list)):
            k1d = gpflow.kernels.Matern32(lengthscales=1e-2, active_dims=[old_dim_list[i]])
            old_kernel += k1d

    ### New kernel
    k2a = gpflow.kernels.Periodic(gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0]), period=1./35.)
    k2b = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0])
    new_kernel = k2a * k2b 
            
    for i in np.arange(len(new_dim_list)):
        k2d = gpflow.kernels.Matern32(lengthscales=1e-2, active_dims=[new_dim_list[i]])
        new_kernel += k2d
    
    kernel_list = [new_kernel, old_kernel]
    dim_list = [new_dim_list, old_dim_list]
    
    return kernel_list, dim_list


def fit_kernel(xtrain, xval, ytrain_tr, yval_tr, kern: gpflow.kernels.Kernel)-> tuple:
    """ Train model and return BIC and MLL for each location"""
        
    # train model
    m = gpflow.models.GPR(data=(xtrain, ytrain_tr.reshape(-1, 1)), kernel=kern)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, options={'maxiter':200})
    gpflow.utilities.print_summary(m)

    # Calculate Bayesian Information Criterion and Mean Log Likelihood
    #bic = me.BIC(m, xval, yval_tr.reshape(-1, 1))
    y_pred,_ = m.predict_y(xval)
    r2 = me.R2(yval_tr, y_pred.numpy())
    print(r2)
    return r2



big_bic_list = []

for i in tqdm.tqdm(range(len(locs))):
    
    xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr, lmbda = dp.point_model(locs[i], maxyear='2020', all_var=True)

    # Locally periodic kernel wrt to time as we want period to be flexible 

    ka = gpflow.kernels.Periodic(gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0]), period=1/35.)
    kb = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0])
    k01 = ka * kb

    kd = gpflow.kernels.Periodic(gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0]), period=1/35.)
    ke = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0])
    k02 = kd * ke

    # Lists
    bic_lists = []
    kernel_list = [k01, k02]
    dim_list = [[], []]

    BIC_lists = []

    for j in range(13):
        
        model_BIC_list = []

        for kern1 in kernel_list:
            #restart_bic = []
            #for r in range(3):
            r2 = fit_kernel(xtrain, xval, ytrain_tr.reshape(-1,1), yval_tr.reshape(-1,1), kern1)
            #    restart_bic.append(bic)
            #avg_bic = np.nanmean(bic)
            model_BIC_list.append(r2)

        best_index = np.argmax(model_BIC_list)
        if j > 0:
            best_dims = dim_list[best_index]
        else:
            best_dims = []
        kernel_list, dim_list = new_kernel_list(best_dims, itera=j)
        BIC_lists.append(model_BIC_list)

    big_bic_list.append(BIC_lists)

    big_bic_arr = np.array(big_bic_list)
    np.save('experiments/slm/slm_bic_arr.npy', big_bic_arr)

# ## Process results

# Averages across locations
bic_arr = np.mean(big_bic_arr, axis=0)
x_arr = np.arange(14)

var_list = ["time", "tcwv", "d2m", "EOF200U1",  "t2m", "EOF850U",  "EOF500U1", "EOF500B2", "EOF200B1",
             "NAO", "EOF500U2", "N34", "EOF850U2", "EOF500B1",]

plt.figure(figsize=(10,5))
plt.scatter(x_arr[0], bic_arr[0,0], marker='o', c='k')
plt.scatter(x_arr[1:], bic_arr[:,0], marker='o', label='New kernel')
plt.scatter(x_arr[1:], bic_arr[:,1], marker='<', label='Best previous kernel')
plt.ylabel('R$^2$')
plt.xlabel('Input features')
plt.xticks(x_arr, labels=var_list, rotation=45)
plt.legend()
plt.savefig('experiments/slm/slm_r2.png', layout='tight')

