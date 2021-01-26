import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
import cartopy.feature as cf
import matplotlib.ticker as tck
import matplotlib.cm as cm
import DataExploration as de

import FileDownloader as fd
import DataPreparation as dp
import DataDownloader as dd


# Open data

data_filepath = fd.update_cds_monthly_data()
mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"


month_labels = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
month_dict = {0: "January", 1: "February", 2: "March", 3: "April", 4: "May", 5: "June", 
              6: "July", 7: "August", 8: "September", 9: "October", 10: "November", 11: "December"} 


def monthly_PDF(data_filepath, mask_filepath, variable="tp", longname=""):

    df = dd.download_data(mask_filepath)
    clean_df = df.dropna()
    df_var = clean_df[["time", variable]]
    reduced_df = df_var.reset_index()
    reduced_df["time"] = (reduced_df["time"] - np.floor(reduced_df["time"])) * 12

    grouped_dfs = []

    for m in np.arange(1,13):
        month_df = reduced_df[reduced_df["time"] == m]
        grouped_dfs.append(month_df[variable])

   # PDF for each month

    """
    fig, axs = plt.subplots(4, 3, sharex=True, sharey=True)

    for i in range(12):
        x= int(i/3)
        y= i%3
        axs[x,y].hist(grouped_dfs[i], bins=50, density=True)
        axs[x,y].set_title(month_dict[i])
        axs[x,y].set_title(month_dict[i])
        axs[x,y].set_xlabel('Total precipation [m]')
        axs[x,y].set_ylabel('Probability density')
        axs[x,y].axvline(np.percentile(grouped_dfs[i], 95), color='k', linestyle='dashed', linewidth=1)
    """

    """
    fig, axs = plt.subplots(12, 1, sharex=True, sharey=True, figsize=(5, 50))

    for i in range(12):
        axs[i].hist(grouped_dfs[i], bins=50, density=True)
        axs[i].set_title(month_dict[i])
        axs[i].set_title(month_dict[i])
        axs[i].set_xlabel('Total precipation [m]')
        axs[i].set_ylabel('Probability density')
        axs[i].axvline(np.percentile(grouped_dfs[i], 95), color='k', linestyle='dashed', linewidth=1)
    """
    
    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)

    for i in range(12):
        x = (i) % 3
        y = int(i / 3)
        axs[x, y].hist(grouped_dfs[i], density=True)
        axs[x, y].set_title(month_dict[i])
        axs[x, y].set_title(month_dict[i])
        axs[x, y].xaxis.set_tick_params(which="both", labelbottom=True)
        axs[x, y].yaxis.set_tick_params(which="both", labelbottom=True)
        axs[x, y].set_xlabel(longname)
        axs[x, y].set_ylabel("Probability density")
        axs[x, y].axvline(
            np.percentile(grouped_dfs[i], 95),
            color="k",
            linestyle="dashed",
            linewidth=1,
            label="95th percentile",
        )

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30))

    plt.show()


def benchmarking_plot(timeseries, y_gpr_t):
    """ Plot probability distribution of model outputs """

    dataframes = []
    
    for ts in timeseries:
        df1 = ts.tp.to_dataframe(name=ts.plot_legend)
        df2 = df1.reset_index()
        df3 = df2.drop(["time", "lon", "lat"], axis=1)
        dataframes.append(df3)

    combined_df = dataframes[0].join(dataframes[1:])
    combined_df["model"] = y_gpr_t[132:312, 0]
    
    time_ds = timeseries[0].time.to_dataframe(name='time')
    months_float = np.ceil((time_ds["time"] - np.floor(time_ds["time"])) * 12)
    df =  combined_df.copy()
    df["time"] = months_float.values
    df["time"] = df["time"].astype(int)

    grouped_dfs = []

    for m in range(1, 13):
        month_df = df[df['time'] == m]
        month_df = month_df.drop(['time'], axis=1)
        grouped_dfs.append(month_df)

    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)
    for i in range(12):
        x = (i) % 3
        y = int(i / 3)
        for b in list(combined_df):
            axs[x, y].hist((grouped_dfs[i])[b], density=True, label=b) #, bins=np.arange(20),alpha=.5)
        axs[x, y].set_title(month_dict[i])
        axs[x, y].set_title(month_dict[i])
        axs[x, y].xaxis.set_tick_params(which="both", labelbottom=True)
        axs[x, y].yaxis.set_tick_params(which="both", labelbottom=True)
        axs[x, y].set_xlabel("Precipitation (mm/day)")
        axs[x, y].set_ylabel("Probability density")
        axs[x, y].axvline(
            np.percentile((grouped_dfs[i])['ERA5'], 95),
            color="k",
            linestyle="dashed",
            linewidth=1,
            label="ERA5 95th percentile",
        )
    plt.legend(loc="upper right")    
    plt.show()
