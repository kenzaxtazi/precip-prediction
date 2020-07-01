# Cross Validation

import os
import calendar
import datetime
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary

import DataDownloader as dd
import Sampling as sa


# Filepaths and URLs
mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'


def kfold_split(x, y, dy=None, folds=5):
    """ 
    Split values into training, validation and test sets.
    The training and validation set are prepared for k folding.

    Inputs
        x: array of features
        y: array of target values
        dy: array of uncertainty on target values
        folds: number of kfolds

    Outputs
        xtrain: feature training set
        xvalidation: feature validation set

        ytrain: target training set
        yvalidation: target validation set

        dy_train: uncertainty training set
        dy_validation: uncertainty validation set
    """
    # Remove test set without shuffling
    
    kf = KFold(n_splits=folds)
    for train_index, test_index in kf.split(x):
        xtrain, xval = x[train_index], x[test_index]
        ytrain, yval = y[train_index], y[test_index]

    return  xtrain, xval, ytrain, yval


def simple_split(x, y):
    xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.30, shuffle=False)
    return  xtrain, xval, ytrain, yval


class GP_estimator(BaseEstimator):

    def __init__(self, kernel, mean_function):
        self.kernel = kernel
        self.mean_function = mean_function
        self.m = 1

    def fit(self, x, y=None, **params):
        self.m = gpflow.models.GPR(data=(x, y.reshape(-1,1)), kernel=self.kernel, mean_function=self.mean_function)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(self.m.training_loss, self.m.trainable_variables, options=dict(maxiter=100))
        # print_summary(self.m)
    
    def predict(self, x):
        y_gpr, y_std = self.m.predict_y(x)
        return y_gpr
    


