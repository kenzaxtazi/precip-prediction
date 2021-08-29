import xarray as xr
import matplotlib.pyplot as plt
import GPy
import pandas as pd
import numpy as np

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

station_dict = {'Arki':[31.154, 76.964, 1176], 'Banjar': [31.65, 77.34, 1914], 'Banjar IMD':[31.637, 77.344, 1427],  
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

station_df = pd.DataFrame.from_dict(station_dict, orient='index',columns=['lat', 'lon', 'elv'])
station_df = station_df.reset_index()

hf_train_df1 = station_df[(station_df['lon']< 77.0) & (station_df['lat']> 32)]
hf_train_df2 = station_df[(station_df['lon']< 76.60) & ((station_df['lat']< 32) & (station_df['lat']> 31.6))]
hf_train_df3 = station_df[(station_df['lon']> 77.0) & (station_df['lat']< 31)]
hf_train_df4 = station_df[(station_df['lon']< 78.0) & (station_df['lon']> 77.0) & (station_df['lat']> 31) & (station_df['lat']< 31.23)]
hf_train_df5 = station_df[(station_df['lon']> 78.2)]
hf_train_stations = list(hf_train_df5['index'].values) + list(hf_train_df2['index'].values) #+ list(hf_train_df3['index'].values) + list(hf_train_df4['index'].values) + list(hf_train_df5['index'].values)
lf_train_stations = hf_train_stations #+ (['Banjar', 'Larji', 'Bhuntar', 'Sainj', 'Bhakra', 'Kasol', 'Suni', 'Pandoh', 'Janjehl', 'Rampur'])
hf_val_stations = hf_train_stations[0]

hf_train_list = []
for station in hf_train_stations:
    station_ds = beas_sutlej_gauges.gauge_download(station, minyear=1980, maxyear=2010)
    station_ds['z'] = station_dict[station][2]
    station_ds['slope'] = srtm.find_slope(station).slope.values
    station_ds = station_ds.set_coords('z')
    station_ds = station_ds.set_coords('slope')
    station_ds = station_ds.expand_dims(dim={'lat': 1, 'lon':1, 'z':1, 'slope':1})
    hf_train_list.append(station_ds)
hf_train_ds = xr.merge(hf_train_list)

# ERA5 data
lf_train_list = []
for station in lf_train_stations:
    station_ds =  era5.gauge_download(station, minyear=1980, maxyear=2010)
    station_ds['z'] = station_dict[station][2]
    station_ds['slope'] = srtm.find_slope(station).slope.values
    station_ds = station_ds.set_coords('z')
    station_ds = station_ds.set_coords('slope')
    station_ds = station_ds.expand_dims(dim={'lat': 1, 'lon':1, 'z':1, 'slope':1})
    lf_train_list.append(station_ds)
lf_train_ds = xr.merge(lf_train_list)

# validation station
val_station_ds = beas_sutlej_gauges.gauge_download(hf_val_stations, minyear=1980, maxyear=2010)
val_station_ds['z'] = station_dict[station][2]
val_station_ds['slope'] = srtm.find_slope(station).slope.values
val_station_ds = station_ds.set_coords('z')
val_station_ds = station_ds.set_coords('slope')
val_station_ds = station_ds.expand_dims(dim={'lat': 1, 'lon':1, 'z':1, 'slope':1})


## Format for model

# To DataFrames
hf_train = hf_train_ds.to_dataframe().dropna().reset_index()
lf_train = lf_train_ds.to_dataframe().dropna().reset_index()
val_df = val_station_ds.to_dataframe().dropna().reset_index()

# Transform data
hf_train['tp_tr'] = dp.log_transform(hf_train['tp'].values)
lf_train['tp_tr'] = dp.log_transform(lf_train['tp'].values)
val_df['tp_tr'] = dp.log_transform(val_df['tp'].values)

# To arrays
# latitude
hf_val_lat = []
lat_list = np.arange(-1, 1, 0.0625)
for lat_ in lat_list:
    station_df = val_df.copy()
    station_df['lat'] += lat_
    arr = station_df[['time', 'lon', 'lat', 'z', 'slope']].values.reshape(-1,5)
    hf_val_lat.append(arr)

# longitude
hf_val_lon = []
lon_list = np.arange(-1, 1, 0.0625)
for lon_ in lon_list:
    station_df = val_df.copy()
    station_df['lon'] += lon_
    arr = station_df[['time', 'lon', 'lat', 'z', 'slope']].values.reshape(-1,5)
    hf_val_lon.append(arr)

# elevation 
hf_val_elv = []
elv_list = np.arange(-1000, 1000, 62.5)
for z_ in elv_list:
    station_df = val_df.copy()
    station_df['z'] += z_
    arr = station_df[['time', 'lon', 'lat', 'z', 'slope']].values.reshape(-1,5)
    hf_val_elv.append(arr)

# slope
hf_val_slope = []
slope_list = np.arange(-1000, 1000, 62.5)
for s in slope_list:
    station_df = val_df.copy()
    station_df['slope'] += s
    arr = station_df[['time', 'lon', 'lat', 'z', 'slope']].values.reshape(-1,5)
    hf_val_slope.append(arr)

# To arrays
hf_x_train = hf_train[['time', 'lon', 'lat', 'z', 'slope']].values.reshape(-1,5)
hf_y_train = hf_train['tp'].values.reshape(-1,1)

lf_y_train_log = lf_train.tp_tr.values.reshape(-1, 1)
hf_y_train_log = hf_train.tp_tr.values.reshape(-1,1)

lf_x_train = lf_train[['time', 'lon', 'lat', 'z', 'slope']].values.reshape(-1,5)
lf_y_train = lf_train['tp'].values.reshape(-1,1)

y_val = val_df['tp'].values.reshape(-1,1)

X_train, Y_train = convert_xy_lists_to_arrays([lf_x_train, hf_x_train], [lf_y_train, hf_y_train])
X_train, Y_train_log = convert_xy_lists_to_arrays([lf_x_train, hf_x_train], [lf_y_train_log, hf_y_train_log])




# Model

def log_linear_mfdgp(X_train, Y_train):
    kernels = [GPy.kern.RBF(5), GPy.kern.RBF(5)]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.optimize_restarts(num_restarts=5)
    return gpy_lin_mf_model

log_lin_mf_model = log_linear_mfdgp(X_train, Y_train_log)


x_met = convert_x_list_to_array([x_val, x_val])
n = x_val.shape[0]

def test_log_likelihood(model, X_test, y_test):
    """ Marginal log likelihood for GPy model on test data"""
    _, test_log_likelihood, _ = model.inference_method.inference(model.kern.rbf_1, X_test, model.likelihood.Gaussian_noise_1, y_test, model.mean_function, model.Y_metadata)
    return test_log_likelihood

lat_heldout_ll = []
for x in hf_val_lat:
    x_met = convert_x_list_to_array([x, x])
    held_log_lik =  test_log_likelihood(log_lin_mf_model, x_met[:len(x)], y_val.reshape(-1,1))
    lat_heldout_ll.append(held_log_lik)

lon_heldout_ll = []
for x in hf_val_lon:
    x_met = convert_x_list_to_array([x, x])
    held_log_lik =  test_log_likelihood(log_lin_mf_model, x_met[:len(x)], y_val.reshape(-1,1))
    lon_heldout_ll.append(held_log_lik)

elv_heldout_ll = []
for x in hf_val_elv:
    x_met = convert_x_list_to_array([x, x])
    held_log_lik =  test_log_likelihood(log_lin_mf_model, x_met[:len(x)], y_val.reshape(-1,1))
    elv_heldout_ll.append(held_log_lik)

slope_heldout_ll = []
for x in hf_val_slope:
    x_met = convert_x_list_to_array([x, x])
    held_log_lik =  test_log_likelihood(log_lin_mf_model, x_met[:len(x)], y_val.reshape(-1,1))
    slope_heldout_ll.append(held_log_lik)

fig, axs = plt.subplots(4)
fig.set_figheight(15)
fig.set_figwidth(15)

axs[0].plot(lat_list, lat_heldout_ll)
axs[0].set_xlabel('latitude')
axs[1].plot(lon_list, lon_heldout_ll)
axs[1].set_xlabel('longitude')
axs[2].plot(elv_list, elv_heldout_ll)
axs[2].set_xlabel('elevation')
axs[3].plot(slope_list, slope_heldout_ll)
axs[3].set_xlabel('slope')

plt.savefig('NLL_vs_distance.png')