import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DataExploration as de
import FileDownloader as fd
import Clustering as cl
import GPModels as gpm

from sklearn.metrics import mean_squared_error, r2_score

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

def plot_predictions(model, x_train, y_train, dy_train, x_test, y_test, dy_test):

    x_plot = np.concatenate((x_train, x_test))
    y_gpr, y_std = model.predict_y(x_plot)
    
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