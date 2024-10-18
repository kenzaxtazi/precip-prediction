#!/usr/bin/env python
# coding: utf-8

# # Whole basin model kernel design
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
import gp.gp_models as gpm
from scipy.special import inv_boxcox

from load import era5, data_dir

tf.random.set_seed(42)

### Load

def new_kernel_list(old_dim_list, itera=int):
    """ Build new kernels """

    N = len(old_dim_list)
    new_dim_list = old_dim_list + [itera+3]

    ### Old kernel
    k1a = gpflow.kernels.Periodic(gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0]), period=1./35.)
    k1b = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0])
    k1c = gpflow.kernels.Matern32(lengthscales=[1e-2,1e-2], active_dims=[1,2])
    old_kernel = k1a * k1b + k1c
    
    if not old_dim_list:
        pass
    else:
        for i in np.arange(len(old_dim_list)):
            k1d = gpflow.kernels.Matern32(lengthscales=1e-2, active_dims=[old_dim_list[i]])
            old_kernel += k1d

    ### New kernel
    k2a = gpflow.kernels.Periodic(gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0]), period=1./35.)
    k2b = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0])
    k2c = gpflow.kernels.Matern32(lengthscales=[1e-2,1e-2], active_dims=[1,2])
    new_kernel = k2a * k2b + k2c
            
    for i in np.arange(len(new_dim_list)):
        k2d = gpflow.kernels.Matern32(lengthscales=1e-2, active_dims=[new_dim_list[i]])
        new_kernel += k2d
    
    kernel_list = [new_kernel, old_kernel]
    dim_list = [new_dim_list, old_dim_list]
    
    return kernel_list, dim_list

def fit_kernel(xtrain, xval, ytrain_tr, yval_tr, kern: gpflow.kernels.Kernel)-> tuple:
    """ Train model and return BIC and MLL for each location"""
        
    # train model
    m = gpflow.models.GPR(data=(xtrain.reshape(-1, 21), ytrain_tr.reshape(-1, 1)), kernel=kern)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, options={'maxiter':200})
    gpflow.utilities.print_summary(m)

    # Calculate Bayesian Information Criterion and Mean Log Likelihood
    #bic = me.BIC(m, xval.reshape(-1, 21), yval_tr.reshape(-1, 1))
    y_pred,_ = m.predict_y(xval)
    r2 = me.R2(yval_tr, y_pred.numpy())
    print(r2)
    return r2


dataset = dp.areal_model_new(location='uib', length=5000, maxyear='2020', all_var=True)
xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr = dataset.sets()


# Locally periodic kernel wrt to time as we want period to be flexible 

ka = gpflow.kernels.Periodic(gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0]), period=1/35.)
kb = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0])
kc = gpflow.kernels.Matern32(lengthscales=[1e-2,1e-2], active_dims=[1,2])
k01 = ka * kb + kc

kd = gpflow.kernels.Periodic(gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0]), period=1/35.)
ke = gpflow.kernels.Matern32(lengthscales=1e-2, variance=1, active_dims=[0])
kf = gpflow.kernels.Matern32(lengthscales=[1e-2,1e-2], active_dims=[1,2])
k02 = kd * ke + kf

# Lists
bic_lists = []
kernel_list = [k01, k02]
dim_list = [[], []]

for j in tqdm.tqdm(range(19)):
    
    model_BIC_list = []

    for kern1 in kernel_list:
        #restart_bic = []
        #for r in range(3):
        r2 = fit_kernel(xtrain, xval, ytrain_tr.reshape(-1,1), yval_tr.reshape(-1,1), kern1)
        #restart_bic.append(bic)
        #avg_bic = np.nanmean(bic)
        model_BIC_list.append(r2)

    best_index = np.argmax(model_BIC_list)
    if j > 0:
        best_dims = dim_list[best_index]
    else:
        best_dims = []
    kernel_list, dim_list = new_kernel_list(best_dims, itera=j)
    bic_lists.append(model_BIC_list)

    bic_arr = np.array(bic_lists)
    np.save('experiments/wbm/wbm_r2.npy', bic_arr)

x_arr = np.arange(19)

var_list = ["time, lon, lat", "tcwv", "slor", "d2m", "z", 
            "EOF200U1", "t2m", "EOF850U1", "EOF500U1", "EOF500B2", 
            "EOF200B1", "anor", "NAO", "EOF500U2", "N34", 
            "EOF850U2", "EOF500B1", "EOF500C1" , "EOF500C2"]

plt.figure(figsize=(15,5))
plt.scatter(x_arr[0], bic_arr[0,0], marker='o', c='k')
plt.scatter(x_arr, bic_arr[:,0], marker='o', label='New kernel')
plt.scatter(x_arr, bic_arr[:,1], marker='<', label='Best previous kernel')
plt.ylabel('Bayesian Information Criterion')
plt.xlabel('Input features')
plt.xticks(x_arr, labels=var_list, rotation=45)
plt.legend()
plt.savefig('experiments/wbm/wbm_bic.png')

