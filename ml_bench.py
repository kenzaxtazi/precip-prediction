"""
Multi Location Model benchmarking

11th August 2021

Part 1: with location only
Part 2: with location and elevation
Part 3: with location elevation and slope
"""

import xarray as xr
import matplotlib.pyplot as plt
import GPy
import pandas as pd

import DataPreparation as dp
from load import beas_sutlej_gauges, era5, srtm

import emukit
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import make_non_linear_kernels, NonLinearMultiFidelityModel

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#### Prepare Data

## Import data

# Gauge data

all_station_dict = {'Arki':[31.154, 76.964, 1176], 'Banjar': [31.65, 77.34, 1914], 'Banjar IMD':[31.637, 77.344, 1427],  
                'Berthin':[31.471, 76.622, 657], 'Bhakra':[31.424, 76.417, 518], 'Barantargh': [31.087, 76.608, 285], 
                'Bharmaur': [32.45, 76.533, 1867], 'Bhoranj':[31.648, 76.698, 834], 'Bhuntar': [31.88, 77.15, 1100], 
                'Churah': [32.833, 76.167, 1358], 'Dadahu':[30.599, 77.437, 635], 'Daslehra': [31.4, 76.55, 561], 
                'Dehra': [31.885, 76.218, 472], 'Dhaula Kuan': [30.517, 77.479, 443], 'Ganguwal': [31.25, 76.486, 345], 
                'Ghanauli': [30.994, 76.527, 284], 'Ghumarwin': [31.436, 76.708, 640], 'Hamirpur': [31.684, 76.519, 763], 
                'Janjehl': [31.52, 77.22, 2071], 'Jogindernagar': [32.036, 76.734, 1442], 'Jubbal':[31.12, 77.67, 2135], 
                'Kalatop': [32.552, 76.018, 2376], 'Kalpa': [31.54, 78.258, 2439], 'Kandaghat': [30.965, 77.119, 1339], 
                'Kangra': [32.103, 76.271, 1318], 'Karsog': [31.383, 77.2, 1417], 'Kasol': [31.357, 76.878, 662], 
                'Kaza': [32.225, 78.072, 3639], 'Kotata': [31.233, 76.534, 320], 'Kothai': [31.119, 77.485, 1531],
                'Kumarsain': [31.317, 77.45, 1617], 'Larji': [31.80, 77.19, 975], 'Lohard': [31.204, 76.561, 290], 
                'Mashobra': [31.13, 77.229, 2240], 'Nadaun': [31.783, 76.35, 480], 'Nahan': [30.559, 77.289, 874], 
                'Naina Devi': [31.279, 76.554, 680], 'Nangal': [31.368, 76.404, 354], 'Olinda': [31.401, 76.385, 363],
                'Pachhad': [30.777, 77.164, 1257], 'Palampur': [32.107, 76.543, 1281], 'Pandoh':[31.67,77.06, 899], 
                'Paonta Sahib': [30.47, 77.625, 433], 'Rakuna': [30.605, 77.473, 688], 'Rampur': [31.454,77.644, 976],
                'Rampur IMD': [31.452, 77.633, 972], 'Rohru':[31.204, 77.751, 1565], 'Sadar-Bilarspur':[31.348, 76.762, 576], 
                'Sadar-Mandi': [31.712, 76.933, 761], 'Sainj': [31.77, 77.31, 1280] , 'Salooni':[32.728, 76.034, 1785],
                'Sarkaghat': [31.704, 76.812, 1155], 'Sujanpur':[31.832, 76.503, 557], 'Sundernargar': [31.534, 76.905, 889], 
                'Suni':[31.238,77.108, 655], 'Suni IMD':[31.23, 77.164, 765], 'Swaghat': [31.713, 76.746, 991], 
                'Theog': [31.124, 77.347, 2101]}

station_df = pd.DataFrame.from_dict(all_station_dict, orient='index',columns=['lat', 'lon', 'elv'])
station_df = station_df.reset_index()
hf_train_df = station_df[(station_df['lon']< 78.0) & ((station_df['lat']> 32) | (station_df['lat']< 31))]
hf_train_stations = list(hf_train_df['index'].values)
lf_train_stations = hf_train_stations + (['Banjar', 'Larji', 'Bhuntar', 'Sainj', 'Bhakra', 'Kasol', 'Suni', 'Pandoh', 'Janjehl', 'Rampur'])
hf_val_stations = ['Banjar', 'Larji', 'Bhuntar', 'Sainj', 'Bhakra', 'Kasol', 'Suni', 'Pandoh', 'Janjehl', 'Rampur']

hf_train_list = []
for station in hf_train_stations:
    station_ds = beas_sutlej_gauges.gauge_download(station, minyear=1980, maxyear=2010)
    station_ds['z'] = all_station_dict[station][2]
    station_ds['slope'] = srtm.find_slope(station).slope.values
    station_ds = station_ds.set_coords('z')
    station_ds = station_ds.set_coords('slope')
    station_ds = station_ds.expand_dims(dim={'lat': 1, 'lon':1, 'z':1, 'slope':1})
    hf_train_list.append(station_ds)
hf_train_ds = xr.merge(hf_train_list)

hf_val_list = []
for station in hf_val_stations:
    station_ds = beas_sutlej_gauges.gauge_download(station, minyear=1980, maxyear=2010)
    station_ds['z'] = all_station_dict[station][2]
    station_ds['slope'] = srtm.find_slope(station).slope.values
    station_ds = station_ds.set_coords('z')
    station_ds = station_ds.set_coords('slope')
    station_ds = station_ds.expand_dims(dim={'lat': 1, 'lon':1, 'z':1, 'slope':1})
    hf_val_list.append(station_ds)
hf_val_ds = xr.merge(hf_val_list)

# ERA5 data
lf_train_list = []
for station in lf_train_stations:
    station_ds =  era5.gauge_download(station, minyear=1980, maxyear=2010)
    station_ds['z'] = all_station_dict[station][2]
    station_ds['slope'] = srtm.find_slope(station).slope.values
    station_ds = station_ds.set_coords('z')
    station_ds = station_ds.set_coords('slope')
    station_ds = station_ds.expand_dims(dim={'lat': 1, 'lon':1, 'z':1, 'slope':1})
    lf_train_list.append(station_ds)
lf_train_ds = xr.merge(lf_train_list)

## Format for model

# To DataFrames
hf_train = hf_train_ds.to_dataframe().dropna().reset_index()
lf_train = lf_train_ds.to_dataframe().dropna().reset_index()
val_df = hf_val_ds.to_dataframe().dropna().reset_index()

# Transform data
hf_train['tp_tr'] = dp.log_transform(hf_train['tp'].values)
lf_train['tp_tr'] = dp.log_transform(lf_train['tp'].values)
val_df['tp_tr'] = dp.log_transform(val_df['tp'].values)

# To arrays
hf_x_train = hf_train[['time', 'lon', 'lat', 'z', 'slope']].values.reshape(-1,5)
hf_y_train = hf_train['tp'].values.reshape(-1,1)

lf_y_train_log = lf_train.tp_tr.values.reshape(-1, 1)
hf_y_train_log = hf_train.tp_tr.values.reshape(-1,1)

lf_x_train = lf_train[['time', 'lon', 'lat', 'z', 'slope']].values.reshape(-1,5)
lf_y_train = lf_train['tp'].values.reshape(-1,1)

x_val = val_df[['time', 'lon', 'lat', 'z', 'slope']].values.reshape(-1,5)
y_val = val_df['tp'].values.reshape(-1,1)

X_train, Y_train = convert_xy_lists_to_arrays([lf_x_train, hf_x_train], [lf_y_train, hf_y_train])
X_train, Y_train_log = convert_xy_lists_to_arrays([lf_x_train, hf_x_train], [lf_y_train_log, hf_y_train_log])


#### Models

# Linear MFDGP 
def linear_mfdgp(X_train, Y_train):
    kernels = [GPy.kern.RBF(5), GPy.kern.RBF(5)]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
    lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)
    lin_mf_model.optimize()
    return lin_mf_model

# Linear MFDGP with log transform
def log_linear_mfdgp(X_train, Y_train):
    kernels = [GPy.kern.RBF(5), GPy.kern.RBF(5)]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    lin_mf_model = GPyMultiOutputWrapper(gpy_lin_mf_model, 2, n_optimization_restarts=5)
    lin_mf_model.optimize()
    return lin_mf_model

# Non-linear MFDGP 
def nonlinear_mfdgp(X_train, Y_train):
    base_kernel = GPy.kern.RBF
    kernels = make_non_linear_kernels(base_kernel, 2, X_train.shape[1] - 1)
    nonlin_mf_model = NonLinearMultiFidelityModel(X_train, Y_train, n_fidelities=2, kernels=kernels, verbose=True, optimization_restarts=5)
    mf1, mf2 = nonlin_mf_model.models
    mf1.Gaussian_noise.variance.fix(0)
    mf2.Gaussian_noise.variance.fix(0)
    nonlin_mf_model.optimize()
    return nonlin_mf_model

# Non-linear MFDGP with log transform
def log_nonlinear_mfdgp(X_train, Y_train):
    base_kernel = GPy.kern.RBF
    kernels = make_non_linear_kernels(base_kernel, 2, X_train.shape[1] - 1)
    nonlin_mf_model = NonLinearMultiFidelityModel(X_train, Y_train, n_fidelities=2, kernels=kernels, verbose=True, optimization_restarts=5)
    mf1, mf2 = nonlin_mf_model.models
    mf1.Gaussian_noise.variance.fix(0)
    mf2.Gaussian_noise.variance.fix(0)
    nonlin_mf_model.optimize()
    return nonlin_mf_model

# Simple GP with log transform
def log_gp(x_train_h, y_train_h):
    kernel = GPy.kern.StdPeriodic(1, period=1) * GPy.kern.RBF(1) + GPy.kern.RBF(5)
    m = GPy.models.GPRegression(x_train_h, y_train_h, kernel)
    m.Gaussian_noise.fix(0)
    m.optimize_restarts(num_restarts=5)
    return m

# Simple GP
def gp(x_train_h, y_train_h):
    kernel = GPy.kern.StdPeriodic(1, period=1) * GPy.kern.RBF(1) + GPy.kern.RBF(5)
    m = GPy.models.GPRegression(x_train_h, y_train_h, kernel)
    m.Gaussian_noise.fix(0)
    m.optimize_restarts(num_restarts=5)
    return m 

# Linear model 
def linear_reg(hf_x_train, hf_y_train):
    linear_m = LinearRegression()
    linear_m.fit(hf_x_train, hf_y_train)
    return linear_m


#### Train models​

lin_mf_model = linear_mfdgp(X_train, Y_train)
log_lin_mf_model = log_linear_mfdgp(X_train, Y_train_log)

nonlin_mf_model = log_nonlinear_mfdgp(X_train, Y_train)
log_nonlin_mf_model = log_nonlinear_mfdgp(X_train, Y_train_log)

gp_m = gp(hf_x_train, hf_y_train)
log_gp_m = log_gp(hf_x_train, hf_y_train_log)

linear_m = linear_reg(hf_x_train, hf_y_train)
log_linear_m = linear_reg(hf_x_train, hf_y_train_log)


#### Model outputs

x_met = convert_x_list_to_array([x_val, x_val])
n = x_val.shape[0]

log_lin_mf_l_y_pred, log_mf_l_y_std_pred = dp.inverse_log_transform(lin_mf_model.predict(x_met[:n]))
log_lin_mf_h_y_pred, log_mf_h_y_std_pred = dp.inverse_log_transform(lin_mf_model.predict(x_met[n:]))

lin_mf_l_y_pred, mf_l_y_std_pred = lin_mf_model.predict(x_met[:n])
lin_mf_h_y_pred, mf_h_y_std_pred = lin_mf_model.predict(x_met[n:])

log_nl_mf_l_y_pred, lo_gmf_l_y_std_pred = dp.inverse_log_transform(log_nonlin_mf_model.predict(x_met[:n]))
log_nl_mf_h_y_pred, log_mf_h_y_std_pred = dp.inverse_log_transform(log_nonlin_mf_model.predict(x_met[n:]))

nl_mf_l_y_pred, mf_l_y_std_pred = nonlin_mf_model.predict(x_met[:n])
nl_mf_h_y_pred, mf_h_y_std_pred = nonlin_mf_model.predict(x_met[n:])

log_gp_y, log_gp_var = dp.inverse_log_transform(log_gp_m.predict(x_val))
gp_y, gp_var = gp_m.predict(x_val)

log_lin_y = dp.inverse_log_transform(log_linear_m.predict(x_val.reshape(-1,5)))
lin_y = linear_m.predict(x_val.reshape(-1,5))


## R2

lin_mf_h_r2 = r2_score(y_val, lin_mf_h_y_pred)
lin_mf_l_r2 = r2_score(y_val, lin_mf_l_y_pred)

nl_mf_h_r2 = r2_score(y_val, nl_mf_h_y_pred)
nl_mf_l_r2 = r2_score(y_val, nl_mf_l_y_pred)
gp_r2 = r2_score(y_val, gp_y)
lin_r2 = r2_score(y_val, lin_y)

log_lin_mf_h_r2 = r2_score(y_val, log_lin_mf_h_y_pred)
log_lin_mf_l_r2 = r2_score(y_val, log_lin_mf_l_y_pred)
log_nl_mf_h_r2 = r2_score(y_val, log_nl_mf_h_y_pred)
log_nl_mf_l_r2 = r2_score(y_val, log_nl_mf_l_y_pred)
log_gp_r2 = r2_score(y_val, log_gp_y)
log_lin_r2 = r2_score(y_val, log_lin_y)


## RMSE

lin_mf_h_rmse = mean_squared_error(y_val, lin_mf_h_y_pred, squared=False)
lin_mf_l_rmse = mean_squared_error(y_val, lin_mf_l_y_pred, squared=False)

nl_mf_h_rmse = mean_squared_error(y_val, nl_mf_h_y_pred, squared=False)
nl_mf_l_rmse = mean_squared_error(y_val, nl_mf_l_y_pred, squared=False)
gp_rmse = mean_squared_error(y_val, gp_y, squared=False)
lin_rmse= mean_squared_error(y_val, lin_y, squared=False)

log_lin_mf_h_rmse = mean_squared_error(y_val, log_lin_mf_h_y_pred, squared=False)
log_lin_mf_l_rmse = mean_squared_error(y_val, log_lin_mf_l_y_pred, squared=False)
log_nl_mf_h_rmse = mean_squared_error(y_val, log_nl_mf_h_y_pred, squared=False)
log_nl_mf_l_rmse = mean_squared_error(y_val, log_nl_mf_l_y_pred, squared=False)
log_gp_rmse = mean_squared_error(y_val, log_gp_y, squared=False)
log_lin_rmse= mean_squared_error(y_val, log_lin_y, squared=False)

print('Log linear MFDGP high R2 = ', lin_mf_h_r2)
print('Log linear MFDGP low R2 = ', lin_mf_l_r2)
print('Log non-linear MFDGP high R2 = ', nl_mf_h_r2)
print('Log non-linear MFDGP low R2 = ', nl_mf_l_r2)
print('Log GP R2 = ', gp_r2)
print('Log linear regression R2 = ', lin_r2)

print('Linear MFDGP high R2 = ', log_lin_mf_h_r2)
print('Linear MFDGP low R2 = ', log_lin_mf_l_r2)
print('Non-linear MFDGP high R2 = ', log_nl_mf_h_r2)
print('Non-linear MFDGP low R2 = ', log_nl_mf_l_r2)
print('GP R2 = ', log_gp_r2)
print('Linear regression R2 = ', log_lin_r2)

print('Log linear MFDGP high RMSE = ', lin_mf_h_rmse)
print('Log inear MFDGP low RMSE = ', lin_mf_l_rmse)
print('Log non-linear MFDGP high RMSE = ',  nl_mf_h_rmse)
print('Log non-linear MFDGP low RMSE = ', nl_mf_l_rmse)
print('Log GP RMSE = ', gp_rmse)
print('Log linear regression RMSE = ', lin_rmse)

print('Linear MFDGP high RMSE = ', log_lin_mf_h_rmse)
print('Linear MFDGP low RMSE = ', log_lin_mf_l_rmse)
print('Non-linear MFDGP high RMSE = ', log_nl_mf_h_rmse)
print('Non-linear MFDGP low RMSE = ', log_nl_mf_l_rmse)
print('GP RMSE = ', log_gp_rmse)
print('Linear regression RMSE = ', log_lin_rmse)