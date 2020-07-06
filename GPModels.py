# GP models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk

import DataExploration as de
import DataPreparation as dp 
import FileDownloader as fd
import CrossValidation as cv
import Metrics as me

import gpflow
import tensorflow as tf
from gpflow.utilities import print_summary


# Filepaths and URLs
    
mask_filepath = 'Data/ERA5_Upper_Indus_mask.nc'
khyber_mask = 'Khyber_mask.nc'
gilgit_mask = 'Gilgit_mask.nc'
ngari_mask = 'Ngari_mask.nc'


'''
# Single point multivariate GP preparation
xtrain, xval, xtest, ytrain, yval, ytest = dp.multivariate_data_prep()

# Single point multivariate GP preparation for cv
xtr, xtest, ytr, ytest = dp.multivariate_cv_data_prep()

# Area multivariate GP preparation
xtrain, xval, xtest, ytrain, yval, ytest = dp.gp_area_prep(khyber_mask)
'''


def area_gp(xtrain, ytrain, xval, yval, save=False):
    """ Returns model and plot of GP model using GPflow for a given area"""

    # model construction
    k1 = gpflow.kernels.Periodic(gpflow.kernels.RBF(lengthscales= 0.3,  variance=6, active_dims=[0]))
    k2 = gpflow.kernels.RBF()
    k = k1 + k2 

    mean_function = gpflow.mean_functions.Linear(A=np.ones((len(xtrain[0]),1)), b=[1])
    m = gpflow.models.GPR(data=(xtrain, ytrain.reshape(-1,1)), kernel=k2, mean_function=mean_function)

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

    x_plot = np.concatenate((xtrain, xval))
    y_plot = np.concatenate((ytrain, yval))
    y_gpr, y_std = m.predict_y(x_plot)

    ## generate 10 samples from posterior
    tf.random.set_seed(1)  # for reproducibility
    samples = m.predict_f_samples(x_plot, 10)  # shape (10, 100, 1)

    ## Take mean accross latitude and longitude

    x_df = pd.DataFrame(x_plot[:, 0:3], columns=['latitude', 'longitude', 'time'])
    y_df = pd.DataFrame(y_plot, columns=['tp true'])
    y_pred_df = pd.DataFrame(np.array(y_gpr), columns=['tp pred'])
    std_pred_df = pd.DataFrame(np.array(y_std), columns=['std pred'])

    results_df = pd.concat([x_df, y_df, y_pred_df, std_pred_df], axis=1)
    plot_df = results_df.groupby(['time']).mean()

    ## Figure 
    '''
    plt.figure()
    plt.title( name + 'Cluster fit')
    # plt.errorbar(x_train[:,0] + 1979, y_train, dy_train, fmt='k.', capsize=2, label='ERA5 training data')
    # plt.errorbar(x_test[:,0] + 1979, y_test, dy_test, fmt='g.', capsize=2, label='ERA5 testing data')
    plt.scatter(plot_df.index + 1979, plot_df['tp true'])
    plt.plot(plot_df.index + 1979, plot_df['tp pred'], color='orange', linestyle='-', label='Prediction')
    plt.fill_between(plot_df.index + 1979, plot_df['tp pred'] - 1.9600 * plot_df['std pred'], plot_df['tp pred'] + 1.9600 * plot_df['std pred'],
             alpha=.2, label='95% confidence interval')
    #plt.plot(x_predictions + 1981, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    plt.legend()
    plt.ylabel('Precipitation [mm/day]')
    plt.xlabel('Year')
    plt.show()
    '''
    print(" {0:.3f} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} |".format(me.R2(m, xtrain, ytrain), me.RMSE(m, xtrain, ytrain), me.R2(m, xval, yval), me.RMSE(m, xval, yval), np.mean(y_gpr), np.mean(y_std)))

    return m


def multi_gp(xtrain, xval, ytrain, yval, save=False):
    """ Returns model and plot of GP model using sklearn """

    # model construction
    k1 = gpflow.kernels.Periodic(gpflow.kernels.RBF(lengthscales=0.3,  variance=6, active_dims=[0]))
    k2 = gpflow.kernels.RBF()
    # k3 = gpflow.kernels.White()
  
    k = k1 + k2  #+k3

    mean_function = gpflow.mean_functions.Linear(A=np.ones((len(xtrain[0]),1)), b=[1])

    m = gpflow.models.GPR(data=(xtrain, ytrain.reshape(-1,1)), kernel=k, mean_function=mean_function)

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
    # print_summary(m)

    x_plot = np.concatenate((xtrain, xval))
    y_gpr, y_std = m.predict_y(x_plot)

    print(" {0:.3f} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} |".format(me.R2(m, xtrain, ytrain), me.RMSE(m, xtrain, ytrain), me.R2(m, xval, yval), me.RMSE(m, xval, yval), np.mean(y_gpr), np.mean(y_std)))
    
    if save == True:
        filepath = save_model(m)
        print(filepath)
        
    return m

def multi_gp_cv(xtr, ytr, save=False, plot=False):

    """" Returns model and plot of GP model using sklearn with cross valiation """

    # model construction
    k1 = gpflow.kernels.Periodic(gpflow.kernels.RBF(lengthscales=0.3,  variance=6, active_dims=[0]))
    k2 = gpflow.kernels.RBF()
    # k3 = gpflow.kernels.White()
  
    k = k1 + k2  #+k3

    mean_function = gpflow.mean_functions.Linear(A=np.ones((len(xtr[0]),1)), b=[1])

    gp_estimator = cv.GP_estimator(kernel=k, mean_function=mean_function)

    scoring = ['r2', 'neg_mean_squared_error']
    score = sk.model_selection.cross_validate(gp_estimator, xtr, ytr, scoring=scoring, cv=sk.model_selection.TimeSeriesSplit())

    print(score['test_r2'].mean())
    print(score['test_neg_mean_squared_error'].mean() * -1)
    
    #y_gpr = sk.model_selection.cross_val_predict(gp_estimator, xtr, ytr) cv=sk.model_selection.TimeSeriesSplit())

    return score


def save_model(model, qualifiers=None): # TODO
    """ Save the model for future use, returns filepath """
    
    frozen_model = gpflow.utilities.freeze(model)
    
    module_to_save = tf.Module()
    predict_fn = tf.function(frozen_model.predict_f, input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)])
    module_to_save.predict = predict_fn

    samples_input = tf.convert_to_tensor(samples_input, dtype=default_float())
    original_result = module_to_save.predict(samples_input)

    filename = 'model_' + now.strftime("%Y-%m-%D-%H-%M-%S") + qualifier

    save_dir = 'Models/' + filename
    tf.saved_model.save(module_to_save, save_dir)

    return save_dir

def restore_model(model_filepath):
    """ Restore a saved model """
    loaded_model = tf.saved_model.load(model_filepath)
    return loaded_model 