#!/usr/bin/env python
# coding: utf-8

# # Stationary model
# 5th February 2024


import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/precip-prediction/')


import tqdm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import utils.metrics as me
import gp.data_prep as dp
import gp.gp_models as gpm
from scipy.special import inv_boxcox

from load import era5, data_dir

xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr, lmbda= dp.areal_model('uib', length=5000)
model = gpm.multi_gp(xtrain, xval, ytrain_tr, yval_tr, lmbda_bc=lmbda, print_perf=False)

yval= inv_boxcox(yval_tr, lmbda)
ytrain = inv_boxcox(ytrain_tr, lmbda)

y_gpr_val0, y_var_val0 = model.predict_y(xval)
y_gpr_val = inv_boxcox(y_gpr_val0, lmbda)

y_gpr_train0, y_var_train0 = model.predict_y(xtrain)
y_gpr_train = inv_boxcox(y_gpr_train0, lmbda)

ytest= inv_boxcox(ytest_tr, lmbda)
y_gpr_test0, y_var_test0 = model.predict_y(xtest)
y_gpr_test = inv_boxcox(y_gpr_test0, lmbda)


r2_val = me.R2(yval, y_gpr_val)
rmse_val= me.RMSE(yval, y_gpr_val)

r2_train = me.R2(ytrain, y_gpr_train)
rmse_train = me.RMSE(ytrain, y_gpr_train)

mll_val = me.MLL(yval_tr, y_gpr_val0, y_var_val0)
mll_train = me.MLL(ytrain_tr, y_gpr_train0, y_var_train0)

r2_test = me.R2(ytest, y_gpr_test)
rmse_test = me.RMSE(ytest, y_gpr_test)
mll_test = me.MLL(ytest_tr, y_gpr_test0, y_var_test0)


# ### Metrics

plt.rcParams.update({'font.size': 15})

plt.figure(figsize=(5, 4))
plt.scatter(ytrain, y_gpr_train, alpha=0.2)
plt.scatter(ytest, y_gpr_test, alpha=0.2)
plt.plot([-1,20], [-1,20], 'k--')
plt.gca().set_aspect('equal')
plt.title('Stationary model')
plt.ylabel('Predicted precipitation [mm/day]')
plt.xlabel('ERA5 precipitation [mm/day]')
plt.savefig('stat_res_new.pdf', bbox_inches='tight')