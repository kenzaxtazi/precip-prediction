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


# Open data

data_filepath = fd.update_cds_monthly_data()
mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"


def monthly_PDF(data_filepath, mask_filepath, variable="tp", longname=""):

    df = dp.download_data(mask_filepath)
    clean_df = df.dropna()
    df_var = clean_df[["time", variable]]
    reduced_df = df_var.reset_index()
    reduced_df["time"] = reduced_df["time"].astype(str)

    month_labels = [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ]
    grouped_dfs = []

    for m in month_labels:
        month_df = reduced_df[reduced_df["time"].str.contains("-" + m + "-")]
        grouped_dfs.append(month_df[variable])

    month_dict = {
        0: "January",
        1: "February",
        2: "March",
        3: "April",
        4: "May",
        5: "June",
        6: "July",
        7: "August",
        8: "September",
        9: "October",
        10: "November",
        11: "December",
    }

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

    plt.show()
