# Metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import DataExploration as de
import DataPreparation as dp
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
    log_RMSE = mean_squared_error(y, y_pred)
    RMSE = dp.inverse_log_transform(log_RMSE) * 1000  # to mm/day
    return np.sqrt(RMSE)


def model_plot(model, location, number=None, posteriors=True, slm=True):
    """ Returns plot for multivariate GP for a single loation """

    if slm is True:
        if number == None:
            xtrain, xval, xtest, ytrain, yval, ytest = dp.point_model(
                location, number=number)
        else:
            xtrain, xval, xtest, ytrain, yval, ytest = dp.point_model(location)
    else:
        if number == None:
            xtrain, xval, xtest, ytrain, yval, ytest = dp.point_model(
                location, number=number)
        else:
            xtrain, xval, xtest, ytrain, yval, ytest = dp.point_model(location)

    xtr = np.concatenate((xtrain, xval), axis=0)
    y_gpr, y_std = model.predict_y(xtr)
    samples = model.predict_f_samples(xtr, 5)

    # to mm/day
    y_gpr_t = dp.inverse_log_transform(y_gpr)
    y_std_t = dp.inverse_log_transform(y_std)
    samples_t = dp.inverse_log_transform(samples)
    ytrain_t = dp.inverse_log_transform(ytrain)
    yval_t = dp.inverse_log_transform(yval)

    plt.figure()

    plt.fill_between(
        xtr[:, 0] + 1979,
        y_gpr_t[:, 0] - 1.9600 * y_std_t[:, 0],
        y_gpr_t[:, 0] + 1.9600 * y_std_t[:, 0],
        alpha=0.5,
        color="lightblue",
        label="95% confidence interval",
    )

    plt.scatter(
        xtrain[:, 0] + 1979, ytrain_t, color="green", label="ERA5 training data", s=10
    )
    plt.scatter(
        xval[:, 0] + 1979, yval_t, color="orange", label="ERA5 validation data", s=10
    )

    plt.plot(xtr[:, 0] + 1979, y_gpr_t, color="C0",
             linestyle="-", label="Prediction")

    if posteriors is True:
        plt.plot(xtr[:, 0] + 1979, samples_t[:, :, 0].T, "C0", linewidth=0.5)

    if location == None:
        plt.title("GP fit")
    else:
        plt.title("GP fit for " +
                  str(location[0]) + "°N " + str(location[1]) + "°E")

    plt.ylabel("Precipitation [mm/day]")
    plt.xlabel("Year")
    plt.legend()

    plt.show()


def ensemble_model_plot(location, models, slm=True):
    """ Returns plot for ensemble of multivariate GP for a single loation """

    palette = sns.color_palette("husl", 10)

    plt.figure()

    for i in range(10):

        if slm is True:
            xtrain, xval, xtest, ytrain, yval, ytest = dp.point_model(
                location, number=i)
        else:
            xtrain, xval, xtest, ytrain, yval, ytest = dp.areal_model_eval(
                location, number=i)

        xtr = np.concatenate((xtrain, xval), axis=0)
        ytr = np.concatenate((ytrain, yval), axis=0)

        log_y_gpr, log_y_std = model.predict_y(xtr)
        log_samples = model.m.predict_f_samples(xtr)

        # to mm/day
        y_gpr_t = dp.inverse_log_transform(log_y_gpr)
        y_std_t = dp.inverse_log_transform(log_y_std)
        samples_t = dp.inverse_log_transform(log_samples)
        ytr_t = dp.inverse_log_transform(ytr)
        #yval_t = dp.inverse_log_transform(yval)

        plt.scatter(xtr[:, 0] + 1979, ytr_t,
                    label="ERA5 data", color=palette[i])
        plt.plot(samples_t)

        plt.plot(
            xtr[:, 0] + 1981, y_gpr_t, color=palette[i], linestyle="-", label="Prediction"
        )
        plt.fill_between(
            xtr[:, 0] + 1981,
            y_gpr_t[:, 0] - 1.9600 * y_std_t[:, 0],
            y_gpr_t[:, 0] + 1.9600 * y_std_t[:, 0],
            alpha=0.2,
            color=palette[i],
            label="95% confidence interval",
        )

    plt.title("GPflow fit")
    plt.ylabel("Precipitation [mm/day]")
    plt.xlabel("Year")
    plt.legend()

    plt.show()


def plot_vs_truth(x_train, y_train, x_test, y_test, m):

    y_test_pred, y_test_std_pred = m.predict_y(x_test)
    y_train_pred, y_train_std_pred = m.predict_y(x_train)

    plt.figure()
    plt.plot([0, 25], [0, 25], linestyle="--", c="black")
    plt.scatter(y_train, y_train_pred, c="blue",
                alpha=0.5, label="Training data")
    plt.scatter(y_test, y_test_pred, c="green",
                alpha=0.5, label="Validation data")
    plt.legend()
    plt.ylabel("Y predicted [mm/day]")
    plt.xlabel("Y true [mm/day]")
    plt.show()


def plot_residuals(x_train, y_train, x_test, y_test, m):

    y_test_pred, y_test_std_pred = m.predict_y(x_test)
    y_train_pred, y_train_std_pred = m.predict_y(x_train)

    # to mm/day
    y_test_t = dp.inverse_log_transform(y_test) * 1000
    y_train_t = dp.inverse_log_transform(y_train) * 1000
    y_test_pred_t = dp.inverse_log_transform(y_test_pred) * 1000
    y_train_pred_t = dp.inverse_log_transform(y_train_pred) * 1000
    y_test_std_t = dp.inverse_log_transform(y_test_std) * 1000
    y_train_std_t = dp.inverse_log_transform(y_train_std) * 1000

    plt.figure()
    plt.title("Residuals")
    plt.plot([1979, 2020], [0, 0], linestyle="--", c="black")
    plt.scatter(
        x_train[:, 2] + 1979,
        y_train_t - np.array(y_train_pred_t).flatten(),
        c="blue",
        alpha=0.5,
        label="ERA5 training data",
    )
    plt.scatter(
        x_test[:, 2] + 1979,
        y_test_t - np.array(y_test_pred_t).flatten(),
        c="green",
        alpha=0.5,
        label="ERA5 validation data",
    )
    plt.legend()
    plt.ylabel("Residuals [mm/day]")
    plt.xlabel("Time")
    plt.show()
