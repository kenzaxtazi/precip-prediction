import GPy
import emukit
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt

from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.convert_lists_to_array \
    import convert_x_list_to_array

from load \
    import beas_sutlej_gauges, era5, srtm, aphrodite, cru, beas_sutlej_wrf


# Import gauge data

station_df = pd.DataFrame.from_csv('_Data/gauge_info.csv')

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

# + list(hf_train_df2['index'].values) + list(hf_train_df3['index'].values)
# + list(hf_train_df4['index'].values) + list(hf_train_df5['index'].values)
hf_train_stations = list(hf_train_df5['index'].values)

hf_train_list = []
for station in hf_train_stations:
    station_ds = beas_sutlej_gauges.gauge_download(
        station, minyear=2000, maxyear=2005)
    station_ds['z'] = station_df[station].values[2]
    station_ds['slope'] = srtm.find_slope(station).slope.values
    station_ds = station_ds.set_coords('z')
    station_ds = station_ds.set_coords('slope')
    station_ds = station_ds.expand_dims(
        dim={'lat': 1, 'lon': 1, 'z': 1, 'slope': 1})
    hf_train_list.append(station_ds)
hf_train_ds = xr.merge(hf_train_list)


hf_train_df = hf_train_ds.to_dataframe().dropna().reset_index()


# Import SRTM data
srtm_ds = xr.open_dataset('Data/SRTM_data.nc')
srtm_ds = srtm_ds.rename({'nlat': 'lat', 'nlon': 'lon'})

# Mask to beas and sutlej
mask_filepath = 'Data/Masks/Beas_Sutlej_highres_mask.nc'
mask = xr.open_dataset(mask_filepath)
mask_da = mask.Overlap
msk_srtm_ds = srtm_ds.where(mask_da > 0, drop=True)

# Import and regrid ERA5 and SRTM data
era5_ds = era5.collect_ERA5([31.885, 76.218], minyear=2000, maxyear=2006)
rg_era5_ds = era5_ds.interp_like(msk_srtm_ds, method='nearest')

rg_srtm_ds = srtm_ds.interp(
    coords={"lon": 76.218, "lat": 31.885, }, method="nearest")

hr_data_ds = xr.merge(
    [rg_era5_ds.tp, msk_srtm_ds.slope, msk_srtm_ds.elevation])
lr_data_ds = xr.merge([era5_ds.tp, rg_srtm_ds.slope, rg_srtm_ds.elevation])


# to DataFrame
hr_data_df = hr_data_ds.to_dataframe().dropna().reset_index()
lr_data_df = lr_data_ds.to_dataframe().dropna().reset_index()

x_train_lf = lr_data_df[['time', 'lat', 'lon']
                        ].values.reshape(-1, 3)  # , 'elevation', 'slope'
y_train_lf = lr_data_df['tp'].values.reshape(-1, 1)

x_train_hf = hf_train_df[['time', 'lon', 'lat']
                         ].values.reshape(-1, 3)  # 'z', 'slope'
y_train_hf = hf_train_df['tp'].values.reshape(-1, 1)

# Input data
X_train = convert_x_list_to_array([x_train_lf, x_train_hf])
Y_train = convert_x_list_to_array([y_train_lf, y_train_hf])

# Plot data
# , 'elevation', 'slope']
x_plot = hr_data_df[['time', 'lat', 'lon']].values.reshape(-1, 3)
X_plot = convert_x_list_to_array([x_plot, x_plot])
n = x_plot.shape[0]


def linear_mfdgp(X_train, Y_train):
    kernels = [GPy.kern.RBF(3), GPy.kern.RBF(3)]
    lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(
        kernels)
    gpy_lin_mf_model = GPyLinearMultiFidelityModel(
        X_train, Y_train, lin_mf_kernel, n_fidelities=2)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise.fix(0)
    gpy_lin_mf_model.mixed_noise.Gaussian_noise_1.fix(0)
    lin_mf_model = GPyMultiOutputWrapper(
        gpy_lin_mf_model, 2, n_optimization_restarts=5)
    lin_mf_model.optimize()
    return lin_mf_model


model = linear_mfdgp(X_train, Y_train)


m = len(x_train_lf)
print(m)
# Predict mean and standard deviation for 10 year
mfdgp_hr_mean, mfdgp_hr_std = model.predict(X_plot[n:])
mfdgp_lr_mean, mfdgp_lr_std = model.predict(X_train[:m])
# To Dataframe
hr_data_df['pred_y'] = mfdgp_hr_mean[:n, 0]
hr_data_df['pred_std'] = mfdgp_hr_std.reshape(-1)

lr_data_df['pred_y'] = mfdgp_lr_mean[:, 0]
lr_data_df['pred_std'] = mfdgp_lr_std.reshape(-1)

# To xarray
reset_df = hr_data_df.reset_index()
multi_index_df = reset_df.set_index(["time", "lat", "lon"])
hr_data_ds = multi_index_df.to_xarray()

reset_df = lr_data_df.reset_index()
multi_index_df = reset_df.set_index(["time", "lat", "lon"])
lr_data_ds = multi_index_df.to_xarray()

# Other datasets
aphro_ds = aphrodite.collect_APHRO(
    [31.885, 76.218], minyear=2000, maxyear=2006)
cru_ds = cru.collect_CRU([31.885, 76.218], minyear=2000, maxyear=2006)
wrf_ds = beas_sutlej_wrf.collect_BC_WRF(
    [31.885, 76.218], minyear=2000, maxyear=2006)

# Differences
aphro_ds['diff'] = aphro_ds.tp - lr_data_ds.pred_y.values.reshape(-1)
cru_ds['diff'] = cru_ds.tp - lr_data_ds.pred_y.values.reshape(-1)
wrf_ds['diff'] = wrf_ds.tp - lr_data_ds.pred_y.values.reshape(-1)


aphro_ds['diff'] = aphro_ds['diff'].expand_dims(
    dim={'lat': 1, 'lon': 1}, axis=[1, 2])
cru_ds['diff'] = cru_ds['diff'].expand_dims(
    dim={'lat': 1, 'lon': 1}, axis=[1, 2])
wrf_ds['diff'] = wrf_ds['diff'].expand_dims(
    dim={'lat': 1, 'lon': 1}, axis=[1, 2])

dataset_list = [wrf_ds.drop('tp'), aphro_ds.drop(
    'tp'), cru_ds.drop(['tp', 'stn'])]

avg_list = []
for ds in dataset_list:
    ds_avg = ds['diff'].mean(dim='time')
    ds_annual_avg = ds_avg

    ds_jun = ds['diff'][5::12]
    ds_jul = ds['diff'][6::12]
    ds_aug = ds['diff'][7::12]
    ds_sep = ds['diff'][8::12]
    ds_monsoon = xr.merge([ds_jun, ds_jul, ds_aug, ds_sep])
    ds_monsoon_avg = ds_monsoon['diff'].mean(dim='time', skipna=True)

    ds_dec = ds['diff'][11::12]
    ds_jan = ds['diff'][0::12]
    ds_feb = ds['diff'][1::12]
    ds_mar = ds['diff'][2::12]
    ds_west = xr.merge([ds_dec, ds_jan, ds_feb, ds_mar])
    ds_west_avg = ds_west['diff'].mean(dim='time', skipna=True)

    ds_avg = xr.concat([ds_annual_avg, ds_monsoon_avg, ds_west_avg], pd.Index(
        ["Annual", "Monsoon (JJAS)", "Winter (DJFM)"], name='t'))

    avg_list.append(ds_avg)


# Dataset list
diff_datasets = xr.concat(avg_list, pd.Index(
    ["BC_WRF-MFDGP", "APHRO-MFDGP", "CRU-MFDGP"], name="Dataset"))

# Prediction plot

a = hr_data_ds.mean(dim='time')

g = a.pred_y.plot(
    x="lon",
    y="lat",
    cbar_kwargs={"label": "Mean prediction (mm/day)"},
    cmap="YlGnBu",
    subplot_kws={"projection": ccrs.PlateCarree()})


g.axes.coastlines()
gl = g.axes.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
g.axes.set_extent([75, 83.5, 29, 34])
g.axes.add_feature(cf.BORDERS)
g.axes.set_xlabel("Longitude")
g.axes.set_ylabel("Latitude")

plt.show()


# Uncertainty plot

g = a.pred_std.plot(
    x="lon",
    y="lat",
    cbar_kwargs={"label": "Standard deviation (mm/day)"},
    cmap="YlGnBu",
    subplot_kws={"projection": ccrs.PlateCarree()})


g.axes.coastlines()
gl = g.axes.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False
g.axes.set_extent([75, 83.5, 29, 34])
g.axes.add_feature(cf.BORDERS)
g.axes.set_xlabel("Longitude")
g.axes.set_ylabel("Latitude")

plt.show()


# Difference plots
g = diff_datasets.plot(
    x="lon",
    y="lat",
    col="t",
    row="Dataset",
    cbar_kwargs={"label": "Total precipitation (mm/day)"},
    cmap="YlGnBu",
    subplot_kws={"projection": ccrs.PlateCarree()})

for ax in g.axes.flat:
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent([75, 83.5, 29, 34])
    ax.add_feature(cf.BORDERS)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.show()
