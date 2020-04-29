
import numpy as np
import matplotlib.pyplot as plt
import PrecipitationDataExploration as pde

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.model_selection import train_test_split


tp_filepath = '/Users/kenzatazi/Downloads/era5_tp_monthly_1979-2019.nc'
tp_ensemble_filepath ='/Users/kenzatazi/Downloads/adaptor.mars.internal-1587987521.7367163-18801-5-5284e7a8-222a-441b-822f-56a2c16614c2.nc'
mpl_filepath = '/Users/kenzatazi/Downloads/era5_msl_monthly_1979-2019.nc'
mask_filepath = '/Users/kenzatazi/Downloads/ERA5_Upper_Indus_mask.nc'

da = pde.apply_mask(tp_filepath, mask_filepath)

# Simple Gaussian Process Model

def simple_gaussian_model(da):
    """ Returns trained model """

    version_da = da.sel(expver=1) 
    std_da = version_da.std(dim='number')
    mean_da = version_da.mean(dim='number')

    gilgit_mean = mean_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    gilgit_std = std_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')

    multi_index_df_mean = gilgit_mean.to_dataframe('Precipitation')
    df_mean= multi_index_df_mean.reset_index()
    df_mean_clean = df_mean.dropna()
    df_mean_clean['time'] = df_mean_clean['time'].astype('int')

    multi_index_df_std = gilgit_std.to_dataframe('Precipitation')
    df_std = multi_index_df_std.reset_index()
    df_std_clean = df_std.dropna()
    df_std_clean['time'] = df_std_clean['time'].astype('int')

    y = df_mean_clean['Precipitation'].values*1000
    dy = df_std_clean['Precipitation'].values*1000

    X_prime = df_mean_clean['time'].values.reshape(-1, 1)
    X = (X_prime - X_prime[0])/ (1e9*60*60*24*365)
    X_train = X[0:400]
    # X_test = X[400:-1]

    y_train = y[0:400]
    dy_train = dy[0:400]
    # y_test = y[400:-1]
    # train_test_split(X, y, test_size=0.5, random_state=42)

    kernel = ExpSineSquared(length_scale=1, periodicity=1)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=dy_train**2, random_state=0).fit(X_train, y_train)
    #print('R2 score = ', gpr.score(X_test, y_test))

    
    X_predictions = np.linspace(0,41,1000)
    X_plot = X_predictions.reshape(-1, 1)
    y_gpr, y_std = gpr.predict(X_plot, return_std=True)
    
    plt.figure()
    plt.title('GP fit for Gilgit (35.8884°N, 74.4584°E, 1500m)')
    plt.errorbar(X_train + 1981, y_train, dy_train, fmt='r.', markersize=10, label='ERA5')
    plt.plot(X_predictions + 1981, y_gpr, 'b-', label='Prediction')
    plt.fill_between(X_predictions + 1981, y_gpr - 1.9600 * y_std, y_gpr + 1.9600 * y_std,
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.legend()
    plt.ylabel('Precipitation [mm/day]')
    plt.xlabel('Year')
    plt.show()




