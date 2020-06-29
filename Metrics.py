# Metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DataExploration as de
import DataPreparation as dp
import FileDownloader as fd
import Clustering as cl
import GPModels as gpm

from sklearn.metrics import mean_squared_error, r2_score

mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'

def R2(model, x, y):
    """ Returns R2 score """
    y_pred, y_std_pred = model.predict_y(x)
    R2 = r2_score(y, y_pred)
    return R2

def RMSE(model, x, y):
    """ Returns RMSE score """
    y_pred, y_std_pred = model.predict_y(x)
    RMSE = mean_squared_error(y, y_pred)
    return RMSE

def plot_predictions(model, mask_filepath, number=None):
    """ Returns plot for multivariate GP for a single loation """
    if number == None:
        da = dp.download_data(mask_filepath, xarray=True)
    else:
        da = dp.download_data(mask_filepath, xarray=True, ensemble=True)
        da = da.sel(number=number).drop('number')

    gilgit = da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    multiindex_df = gilgit.to_dataframe()
    df_clean = multiindex_df.reset_index()
    df = df_clean.drop(columns=['latitude', 'longitude'])

    df['time'] = df['time'].astype('int')
    df['time'] = (df['time'] - df['time'].min())/ (1e9*60*60*24*365)
    df['tp'] = df['tp']*1000  # to mm
    df_clean = df.dropna()

    # Training and validation data
    tr_df = df_clean[ df_clean['time']< df_clean['time'].max()*0.9]
    xtr = tr_df.drop(columns=['tp']).values
    ytr = tr_df['tp'].values

    y_gpr, y_std = model.predict_y(xtr)
    samples = model.predict_f_samples(xtr, 5)
    
    plt.figure()
    plt.title('GPflow fit for Gilgit (35.8884°N, 74.4584°E, 1500m)')
    plt.scatter(xtr[:,0] + 1981, ytr, label='ERA5 data')
    plt.plot(xtr[:,0] + 1981, y_gpr, color='orange', linestyle='-', label='Prediction')
    plt.fill_between(xtr[:,0]+ 1981, y_gpr[:, 0] - 1.9600 * y_std[:, 0], y_gpr[:, 0] + 1.9600 * y_std[:, 0],
             alpha=.5, color='lightblue', label='95% confidence interval')
    plt.plot(xtr[:,0] + 1981, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    plt.legend()
    plt.ylabel('Precipitation [mm/day]')
    plt.xlabel('Year')
    plt.show()

def plot_vs_truth(x_train, y_train, x_test, y_test, m):
    
    y_test_pred, y_test_std_pred = m.predict_y(x_test)
    y_train_pred, y_train_std_pred = m.predict_y(x_train)

    plt.figure()
    plt.plot([0,8],[0,8], linestyle='--', c='black')
    plt.scatter(y_train, y_train_pred, c='blue', alpha=0.5, label='ERA5 training data')
    plt.scatter(y_test, y_test_pred, c='green', label='ERA5 validation data')
    plt.legend()
    plt.ylabel('Y predicted')
    plt.xlabel('Y true')
    plt.show() 

def plot_residuals(x_train, y_train, x_test, y_test, m):
    
    y_test_pred, y_test_std_pred = m.predict_y(x_test)
    y_train_pred, y_train_std_pred = m.predict_y(x_train)

    plt.figure()
    plt.title('Residuals')
    plt.plot([1979, 2020],[0,0], linestyle='--', c='black')
    plt.scatter(x_train[:,0] + 1979, y_train-np.array(y_train_pred).flatten(), c='blue', alpha=0.5, label='ERA5 training data')
    plt.scatter(x_test[:,0] + 1979, y_test-np.array(y_test_pred).flatten(), c='green', alpha=0.5, label='ERA5 validation data')
    plt.legend()
    plt.ylabel('Residuals')
    plt.xlabel('Time')
    plt.show() 