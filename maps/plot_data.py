# Collection of map plotting functions

import calendar

import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from load import era5, cru, beas_sutlej_wrf, gpm, aphrodite, location_sel


def annual_map(data_filepath, mask_filepath, variable, year, cumulative=False):
    """ Annual map """

    da = location_sel.apply_mask(data_filepath, mask_filepath)
    ds_year = da.sel(time=slice(str(year) + "-01-16T12:00:00",
                     str(year + 1) + "-01-01T12:00:00"))
    ds_var = ds_year[variable] * 1000  # to mm/day

    if cumulative is True:
        ds_processed = cumulative_monthly(ds_var)
        ds_final = ds_processed.sum(dim="time")
    else:
        ds_final = ds_var.std(dim="time")  # TODO weighted mean

    print(ds_final)

    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    g = ds_final["tp_0001"].plot(cmap="magma_r", vmin=0.001, cbar_kwargs={
        "label": "Precipitation standard deviation [mm/day]",
        "extend": "neither", "pad": 0.10})
    g.cmap.set_under("white")
    # ax.add_feature(cf.BORDERS)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.title("Upper Indus Basin Total Precipitation " + str(year) + "\n \n")
    plt.show()


def change_maps(data_filepath, mask_filepath, variable):
    """ Average annual change from 1979 to 1989, 1999, 2009 and 2019 """

    da = location_sel.apply_mask(data_filepath, mask_filepath)
    da_var = da[variable] * 1000  # to mm/day

    da_1979 = da_var.sel(
        time=slice("1979-01-16T12:00:00", str(+ 1) + "1980-01-01T12:00:00")
    )
    da_processed = cumulative_monthly(da_1979)
    basin_1979_sum = da_processed.sum(dim="time")

    basin_1989 = da_var.sel(time=slice(
        "1989-01-01T12:00:00", "1990-01-01T12:00:00"))
    basin_1989_sum = cumulative_monthly(basin_1989).sum(dim="time")
    basin_1989_change = basin_1989_sum / basin_1979_sum - 1

    basin_1999 = da_var.sel(time=slice(
        "1999-01-01T12:00:00", "2000-01-01T12:00:00"))
    basin_1999_sum = cumulative_monthly(basin_1999).sum(dim="time")
    basin_1999_change = basin_1999_sum / basin_1979_sum - 1

    basin_2009 = da_var.sel(time=slice(
        "2009-01-01T12:00:00", "2010-01-01T12:00:00"))
    basin_2009_sum = cumulative_monthly(basin_2009).sum(dim="time")
    basin_2009_change = basin_2009_sum / basin_1979_sum - 1

    basin_2019 = da_var.sel(time=slice(
        "2019-01-01T12:00:00", "2020-01-01T12:00:00"))
    basin_2019_sum = cumulative_monthly(basin_2019).sum(dim="time")
    basin_2019_change = basin_2019_sum / basin_1979_sum - 1

    basin_changes = xr.concat(
        [basin_1989_change, basin_1999_change,
            basin_2009_change, basin_2019_change],
        pd.Index(["1989", "1999", "2009", "2019"], name="year"),
    )

    g = basin_changes.plot(
        x="longitude",
        y="latitude",
        col="year",
        col_wrap=2,
        subplot_kws={"projection": ccrs.PlateCarree()},
        cbar_kwargs={
            "label": "Precipitation change",
            "format": tck.PercentFormatter(xmax=1.0),
        },
    )

    for ax in g.axes.flat:
        ax.coastlines()
        ax.gridlines()
        ax.set_extent([71, 83, 30, 38])
        ax.add_feature(cf.BORDERS)
        ax.gridlines(draw_labels=True)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.show()


def cumulative_monthly(da):
    """ Multiplies monthly averages by the number of day in each month """
    x, y, z = np.shape(da.values)
    times = np.datetime_as_string(da.time.values)
    days_in_month = []
    for t in times:
        year = t[0:4]
        month = t[5:7]
        days = calendar.monthrange(int(year), int(month))[1]
        days_in_month.append(days)
    dim = np.array(days_in_month)
    dim_mesh = np.repeat(dim, y * z).reshape(x, y, z)

    return da * dim_mesh


def multi_dataset_map(location, seasonal=False):
    """
    Create maps of raw values from multiple datasets
    - ERA5
    - GPM PR
    - APHRODITE
    - CRU
    - Bannister corrected WRF
    """

    # Load data
    aphro_ds = aphrodite.collect_APHRO(location, minyear=2000, maxyear=2011)
    cru_ds = cru.collect_CRU(location, minyear=2000, maxyear=2011)
    era5_ds = era5.collect_ERA5(location, minyear=2000, maxyear=2011)
    gpm_ds = gpm.collect_GPM(location,  minyear=2000, maxyear=2011)
    wrf_ds = beas_sutlej_wrf.collect_BC_WRF(
        location, minyear=2000, maxyear=2011)

    dataset_list = [gpm_ds, era5_ds, wrf_ds, aphro_ds, cru_ds]

    # Slice and take averages
    avg_list = []
    for ds in dataset_list:
        ds_avg = ds.tp.mean(dim='time')

        if seasonal is True:
            ds_annual_avg = ds_avg

            ds_jun = ds.tp[5::12]
            ds_jul = ds.tp[6::12]
            ds_aug = ds.tp[7::12]
            ds_sep = ds.tp[8::12]
            ds_monsoon = xr.merge([ds_jun, ds_jul, ds_aug, ds_sep])
            ds_monsoon_avg = ds_monsoon.tp.mean(dim='time')

            ds_dec = ds.tp[11::12]
            ds_jan = ds.tp[0::12]
            ds_feb = ds.tp[1::12]
            ds_mar = ds.tp[2::12]
            ds_west = xr.merge([ds_dec, ds_jan, ds_feb, ds_mar])
            ds_west_avg = ds_west.tp.mean(dim='time')

            ds_avg = xr.concat([ds_annual_avg, ds_monsoon_avg, ds_west_avg],
                               pd.Index(["Annual", "Monsoon (JJAS)",
                                         "Winter (DJFM)"], name='t'))

        avg_list.append(ds_avg)

    datasets = xr.concat(avg_list, pd.Index(
        ["TRMM", "ERA5", "BC_WRF", "APHRO", "CRU"], name="Dataset"))

    # Plot maps

    g = datasets.plot(
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
