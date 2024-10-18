import gpflow
import sys
sys.path.append('/data/hpcdata/users/kenzi22/')
sys.path.append('/data/hpcdata/users/kenzi22/precip-prediction')
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
from scipy.special import inv_boxcox

import utils.metrics as me
import gp.data_prep as dp
from load import era5
import pickle
import gp.gp_models as gpm

# ## Load 
data = era5.collect_ERA5('uib', minyear='1970', maxyear='1980', all_var=True)

df = data.to_dataframe()
loc_df = df.groupby(['lat','lon']).mean().reset_index()
loc_df.dropna(inplace=True)
locs = loc_df[['lon','lat']].values

tf.random.set_seed(42)

r2_train_list = []
rmse_train_list = []
mll_train_list = []

r2_val_list = []
rmse_val_list = []
mll_val_list = []

r2_test_list = []
rmse_test_list = []
mll_test_list = []

for i in tqdm.tqdm(range(len(locs))):
    
    dataset = dp.point_model(locs[i], maxyear='2020', all_var=False)
    xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr = dataset.sets()
    lmbda = dataset.l
    yscaler = dataset.yscaler

    m = gpm.multi_gp(xtrain, xval, ytrain_tr, yval_tr, lmbda, yscaler, kern="point", save=False, print_perf=False)

    yval= inv_boxcox(yval_tr, lmbda)
    ytest= inv_boxcox(ytest_tr, lmbda)
    ytrain= inv_boxcox(ytrain_tr, lmbda)

    y_gpr_train0, y_var_train0 = m.predict_y(xtrain)
    y_gpr_train = inv_boxcox(y_gpr_train0, lmbda)

    y_gpr_val0, y_var_val0 = m.predict_y(xval)
    y_gpr_val = inv_boxcox(y_gpr_val0, lmbda)

    y_gpr_test0, y_var_test0 = m.predict_y(xtest)
    y_gpr_test = inv_boxcox(y_gpr_test0, lmbda)

    r2_train_list.append(me.R2(ytrain, y_gpr_train))
    rmse_train_list.append(me.RMSE(ytrain, y_gpr_train))
    mll_train_list.append(me.MLL(ytrain_tr, y_gpr_train0, y_var_train0))

    r2_val_list.append(me.R2(yval, y_gpr_val))
    rmse_val_list.append(me.RMSE(yval, y_gpr_val))
    mll_val_list.append(me.MLL(yval_tr, y_gpr_val0, y_var_val0))

    r2_test_list.append(me.R2(ytest, y_gpr_test))
    rmse_test_list.append(me.RMSE(ytest, y_gpr_test))
    mll_test_list.append(me.MLL(ytest_tr, y_gpr_test0, y_var_test0))

r2_val = np.nanmean(np.array(r2_val_list))
rmse_val = np.nanmean(np.array(rmse_val_list))
mll_val = np.nanmean(np.array(mll_val_list))

print('Train R2: ', r2_val)
print('Train RMSE: ', rmse_val)
print('Train MLL: ', mll_val)
