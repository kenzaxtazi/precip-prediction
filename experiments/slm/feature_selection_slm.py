#!/usr/bin/env python
# coding: utf-8

# # Building kernel
# 16th April 2024

import gpflow
import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/precip-prediction/')

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


def new_kernel_list(old_kernel, iter=int):
    """ Build new kernels """

    new_kernel = gpflow.kernels.Matern32(lengthscales=1, variance=1, active_dims=[iter+1])

    k1 = old_kernel + new_kernel
    #k2 = old_kernel * new_kernel
    k3 = old_kernel

    kernel_list = [k1, k3] #k2
    
    return kernel_list

def fit_kernel(xtrain, xval, ytrain_tr, yval_tr, kernel: gpflow.kernels.Kernel)-> tuple:
    """ Train model and return BIC and MLL for each location"""
        
    # train model
    m = gpflow.models.GPR(data=(xtrain, ytrain_tr.reshape(-1, 1)), kernel=kernel)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables)

    # Calculate Bayesian Information Criterion and Mean Log Likelihood
    bic = me.BIC(m, xval, yval_tr.reshape(-1, 1))
    return bic



big_bic_list = []

for i in tqdm.tqdm(range(len(locs))):

    try:
        xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr = dp.point_model(locs[i], maxyear='2020')

        # Locally periodic kernel wrt to time as we want period to be flexible 

        k1 = gpflow.kernels.Periodic(gpflow.kernels.Matern32(lengthscales=1, variance=1, active_dims=[0]))
        k1a = gpflow.kernels.Matern32(lengthscales=1, variance=1, active_dims=[0])
        ka = k1 * k1a

        kernel_list = [ka]

        BIC_lists = []

        for j in range(13):
            
            model_BIC_list = []

            for k in kernel_list:
                bic = fit_kernel(xtrain, xval, ytrain_tr, yval_tr, k)
                model_BIC_list.append(bic)

            best_index = np.argmin(model_BIC_list)
            best_kernel = kernel_list[best_index]
            kernel_list = new_kernel_list(best_kernel, iter=j)
            BIC_lists.append(model_BIC_list)

        big_bic_list.append(BIC_lists)

        with open("wbm_bic_list", "wb") as fp:   #Pickling
            pickle.dump(big_bic_list, fp)

    except:
        print('Error at location: ', locs[i])
        pass

    


# ## Process results

# Averages across locations
mean_init_bic = np.mean(big_bic_list[:, 0], axis=0)
BIC_arr = np.mean(np.array(big_bic_list[:, 1:]), axis=0)
x = np.arange(1, 14)


plt.figure(figsize=(7,5))
plt.scatter(0, BIC_lists[0], marker='o', c='k')
plt.scatter(x, BIC_arr[:,0], marker='o', label='additive kernel')
plt.scatter(x, BIC_arr[:,1], marker='<', label='best previous kernel')
plt.ylabel('Bayesian Information Criterion')
plt.xlabel('Kernel design iterations')
plt.xticks(np.arange(0, 13, 1), labels=var_list)
plt.legend()
plt.savefig('slm_bic.png')

