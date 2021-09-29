
import xarray as xr
import matplotlib.pyplot as plt
import GPy
import pandas as pd

import gp.data_prep as dp
from load import beas_sutlej_gauges, era5, srtm

import emukit
from emukit.multi_fidelity.convert_lists_to_array \
    import convert_x_list_to_array, convert_xy_lists_to_arrays
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel

# Prepare Data

# Import data

# Gauge data

station_df = pd.DataFrame.from_csv('Data/gauge_info.csv')

hf_train_df1 = station_df[(station_df['lon'] < 77.0)
                          & (station_df['lat'] > 32)]
hf_train_df2 = station_df[(station_df['lon'] < 76.60) & (
    (station_df['lat'] < 32) & (station_df['lat'] > 31.6))]
hf_train_df3 = station_df[(station_df['lon'] > 77.0)
                          & (station_df['lat'] < 31)]
hf_train_df4 = station_df[(station_df['lon'] < 78.0) & (
    station_df['lon'] > 77.0) & (station_df['lat'] > 31)
    & (station_df['lat'] < 31.23)]
hf_train_df5 = station_df[(station_df['lon'] > 78.2)]
hf_train_stations = list(hf_train_df1['index'].values)
+ list(hf_train_df2['index'].values)
+ list(hf_train_df3['index'].values)
+ list(hf_train_df4['index'].values)
+ list(hf_train_df5['index'].values)

lf_train_stations = hf_train_stations + \
    (['Banjar', 'Larji', 'Bhuntar', 'Sainj', 'Bhakra',
     'Kasol', 'Suni', 'Pandoh', 'Janjehl', 'Rampur'])
hf_val_stations = ['Banjar', 'Larji', 'Bhuntar', 'Sainj',
                   'Bhakra', 'Kasol', 'Suni', 'Pandoh', 'Janjehl', 'Rampur']

hf_train_list = []
for station in hf_train_stations:
    station_ds = beas_sutlej_gauges.gauge_download(
        station, minyear=1980, maxyear=2010)
    station_ds['z'] = station_df[station].values[2]
    station_ds['slope'] = srtm.find_slope(station).slope.values
    station_ds = station_ds.set_coords('z')
    station_ds = station_ds.set_coords('slope')
    station_ds = station_ds.expand_dims(
        dim={'lat': 1, 'lon': 1, 'z': 1, 'slope': 1})
    hf_train_list.append(station_ds)
hf_train_ds = xr.merge(hf_train_list)

hf_val_list = []
for station in hf_val_stations:
    station_ds = beas_sutlej_gauges.gauge_download(
        station, minyear=1980, maxyear=2010)
    station_ds['z'] = station_df[station].values[2]
    station_ds['slope'] = srtm.find_slope(station).slope.values
    station_ds = station_ds.set_coords('z')
    station_ds = station_ds.set_coords('slope')
    station_ds = station_ds.expand_dims(
        dim={'lat': 1, 'lon': 1, 'z': 1, 'slope': 1})
    hf_val_list.append(station_ds)
hf_val_ds = xr.merge(hf_val_list)

# ERA5 data
lf_train_list = []
for station in lf_train_stations:
    station_ds = era5.gauge_download(station, minyear=1980, maxyear=2010)
    station_ds['z'] = station_df[station].values[2]
    station_ds['slope'] = srtm.find_slope(station).slope.values
    station_ds = station_ds.set_coords('z')
    station_ds = station_ds.set_coords('slope')
    station_ds = station_ds.expand_dims(
        dim={'lat': 1, 'lon': 1, 'z': 1, 'slope': 1})
    lf_train_list.append(station_ds)
lf_train_ds = xr.merge(lf_train_list)

# Format for model

# To DataFrames
hf_train = hf_train_ds.to_dataframe().dropna().reset_index()
lf_train = lf_train_ds.to_dataframe().dropna().reset_index()
val_df = hf_val_ds.to_dataframe().dropna().reset_index()

# Transform data
hf_train['tp_tr'] = dp.log_transform(hf_train['tp'].values)
lf_train['tp_tr'] = dp.log_transform(lf_train['tp'].values)
val_df['tp_tr'] = dp.log_transform(val_df['tp'].values)

# To arrays
hf_x_train = hf_train[['time', 'lon', 'lat',
                       'z', 'slope']].values.reshape(-1, 5)
hf_y_train = hf_train['tp'].values.reshape(-1, 1)

lf_y_train_log = lf_train.tp_tr.values.reshape(-1, 1)
hf_y_train_log = hf_train.tp_tr.values.reshape(-1, 1)

lf_x_train = lf_train[['time', 'lon', 'lat',
                       'z', 'slope']].values.reshape(-1, 5)
lf_y_train = lf_train['tp'].values.reshape(-1, 1)

x_val = val_df[['time', 'lon', 'lat', 'z', 'slope']].values.reshape(-1, 5)
y_val = val_df['tp'].values.reshape(-1, 1)

X_train, Y_train = convert_xy_lists_to_arrays(
    [lf_x_train, hf_x_train], [lf_y_train, hf_y_train])
X_train, Y_train_log = convert_xy_lists_to_arrays(
    [lf_x_train, hf_x_train], [lf_y_train_log, hf_y_train_log])


def log_linear_mfdgp(X_train, Y_train):
    kernels = [GPy.kern.RBF(5), GPy.kern.RBF(5)]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(
        kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(
        X_train, Y_train, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    lin_mf_model = GPyMultiOutputWrapper(
        gpy_lin_mf_model, 2, n_optimization_restarts=5)
    lin_mf_model.optimize()
    return lin_mf_model


log_lin_mf_model = log_linear_mfdgp(X_train, Y_train_log)

# Model outputs
x_met = convert_x_list_to_array([x_val, x_val])
n = x_val.shape[0]

log_lin_mf_l_y_pred, log_mf_l_y_std_pred = dp.inverse_log_transform(
    log_lin_mf_model.predict(x_met[:n]))
log_lin_mf_h_y_pred, log_mf_h_y_std_pred = dp.inverse_log_transform(
    log_lin_mf_model.predict(x_met[n:]))

# Plots
difference = log_lin_mf_l_y_pred - log_lin_mf_h_y_pred
val_df['diff'] = difference

plt.figure()
plt.scatter(val_df['time'], val_df['diff'])
plt.ylabel('Mapping function')
plt.xlabel('Time')
plt.savefig('mapping_function_time.png')

plt.figure()
plt.scatter(val_df['lat'], val_df['diff'])
plt.ylabel('Mapping function')
plt.xlabel('Latitude')
plt.savefig('mapping_function_lat.png')

plt.figure()
plt.scatter(val_df['lon'], val_df['diff'])
plt.ylabel('Mapping function')
plt.xlabel('Longitude')
plt.savefig('mapping_function_lon.png')

plt.figure()
plt.scatter(val_df['slope'], val_df['diff'])
plt.ylabel('Mapping function')
plt.xlabel('Slope')
plt.savefig('mapping_function_slope.png')

plt.figure()
plt.scatter(val_df['z'], val_df['diff'])
plt.ylabel('Mapping function')
plt.xlabel('Elevation')
plt.savefig('mapping_function_elev.png')
