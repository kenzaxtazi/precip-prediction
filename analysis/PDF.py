import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

from load import era5


# Open data
# data_filepath = fd.update_cds_monthly_data()
mask_filepath = "Data/ERA5_Upper_Indus_mask.nc"


month_labels = ["01", "02", "03", "04", "05",
                "06", "07", "08", "09", "10", "11", "12"]
month_dict = {0: "January", 1: "February", 2: "March", 3: "April", 4: "May",
              5: "June", 6: "July", 7: "August", 8: "September",
              9: "October", 10: "November", 11: "December"}


def monthly_PDF(timeseries, variable="tp", longname=""):

    combined_df = pd.DataFrame()

    for ts in timeseries:
        df1 = ts.tp.to_dataframe(name=ts.plot_legend)
        df2 = df1.dropna().reset_index()
        df3 = df2.drop(["time", "lon", "lat"], axis=1)
        combined_df[ts.plot_legend] = df3[ts.plot_legend]

    df = era5.download_data(mask_filepath)
    clean_df = df.dropna()
    df_var = clean_df[["time", variable]]
    reduced_df = df_var.reset_index()
    reduced_df["time"] = (reduced_df["time"] -
                          np.floor(reduced_df["time"])) * 12

    grouped_dfs = []

    for m in np.arange(1, 13):
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
        axs[x,y].axvline(np.percentile(grouped_dfs[i], 95), color='k',
        linestyle='dashed', linewidth=1)
    """

    """
    fig, axs = plt.subplots(12, 1, sharex=True, sharey=True, figsize=(5, 50))

    for i in range(12):
        axs[i].hist(grouped_dfs[i], bins=50, density=True)
        axs[i].set_title(month_dict[i])
        axs[i].set_title(month_dict[i])
        axs[i].set_xlabel('Total precipation [m]')
        axs[i].set_ylabel('Probability density')
        axs[i].axvline(np.percentile(grouped_dfs[i], 95), color='k',
        linestyle='dashed', linewidth=1)
    """

    _fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)

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


def benchmarking_plot(timeseries, kernel_density=False):
    """ Plot probability distribution of model outputs """

    combined_df = pd.DataFrame()

    for ts in timeseries:
        df1 = ts.tp.to_dataframe(name=ts.plot_legend)
        df2 = df1.reset_index()
        df3 = df2.drop(["time", "lon", "lat"], axis=1)
        print(df3.shape)
        combined_df[ts.plot_legend] = df3[ts.plot_legend]

    time_ds = timeseries[0].time.to_dataframe(name='time')
    months_float = np.ceil((time_ds["time"] - np.floor(time_ds["time"])) * 12)
    df = combined_df.copy()
    df["time"] = months_float.values
    df["time"] = df["time"].astype(int)

    grouped_dfs = []

    for m in range(1, 13):
        month_df = df[df['time'] == m]
        month_df = month_df.drop(['time'], axis=1)
        grouped_dfs.append(month_df)

    # Plot
    _fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)

    for i in range(12):
        x = (i) % 3
        y = int(i / 3)
        for b in list(combined_df):
            data = grouped_dfs[i][b].values
            X_plot = np.linspace(0, data.max(), 1000)[:, np.newaxis]
            if kernel_density is True:
                bandwidth = np.arange(0.05, 2, .05)
                X = data.reshape(-1, 1)
                kde = KernelDensity(kernel='gaussian')
                grid = GridSearchCV(kde, {'bandwidth': bandwidth})
                grid.fit(X)
                kde = grid.best_estimator_
                log_dens = kde.score_samples(X_plot)
                axs[x, y].fill(X_plot[:, 0], np.exp(log_dens), label=b)
            else:
                axs[x, y].hist((grouped_dfs[i])[b], density=True, label=b,
                               bins=np.arange(0, int(combined_df.max().max())))
                # alpha=.5)
        axs[x, y].set_title(month_dict[i])
        axs[x, y].set_title(month_dict[i])
        axs[x, y].xaxis.set_tick_params(which="both", labelbottom=True)
        axs[x, y].yaxis.set_tick_params(which="both", labelbottom=True)
        axs[x, y].set_xlabel("Precipitation (mm/day)")
        axs[x, y].set_ylabel("Probability density")

        '''
        axs[x, y].axvline(
            np.percentile((grouped_dfs[i])['ERA5'], 95),
            color="k",
            linestyle="dashed",
            linewidth=1,
            label="ERA5 95th percentile",
        )
        '''
    plt.legend(loc="upper right")
    plt.show()


def cdf_benchmarking_plot(timeseries, kernel_density=False):
    """ Plot probability distribution of model outputs """

    combined_df = pd.DataFrame()

    for ts in timeseries:
        df1 = ts.tp.to_dataframe(name=ts.plot_legend)
        df2 = df1.dropna().reset_index()
        df3 = df2.drop(["time", "lon", "lat"], axis=1)
        combined_df[ts.plot_legend] = df3[ts.plot_legend]

    time_ds = timeseries[0].time.to_dataframe(name='time')
    months_float = np.ceil((time_ds["time"] - np.floor(time_ds["time"])) * 12)
    df = combined_df.copy()
    df["time"] = months_float.values
    df["time"] = df["time"].astype(int)

    grouped_dfs = []

    for m in range(1, 13):
        month_df = df[df['time'] == m]
        month_df = month_df.drop(['time'], axis=1)
        grouped_dfs.append(month_df)

    # Plot
    _fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)

    for i in range(12):
        x = (i) % 3
        y = int(i / 3)
        for b in list(combined_df):

            data = grouped_dfs[i][b].values
            X_plot = np.linspace(0, data.max(), 1000)[:, np.newaxis]
            # data_sorted = np.sort(data)
            # p = 1. * np.arange(len(data)) / (len(data) - 1)

            if kernel_density is True:
                n, _bins, _patches = np.hist(
                    (grouped_dfs[i])[b], cumulative=True, density=True,
                    label=b)
                X = n.reshape(-1, 1)
                kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)
                log_dens = kde.score_samples(X_plot)
                axs[x, y].plot(X_plot[:, 0], np.exp(log_dens), label=b)

            else:
                axs[x, y].hist((grouped_dfs[i])[b],
                               cumulative=True, density=True, label=b)

        axs[x, y].set_title(month_dict[i])
        axs[x, y].set_title(month_dict[i])
        axs[x, y].xaxis.set_tick_params(which="both", labelbottom=True)
        axs[x, y].yaxis.set_tick_params(which="both", labelbottom=True)
        axs[x, y].set_xlabel("Precipitation (mm/day)")
        axs[x, y].set_ylabel("CDF")

        axs[x, y].axvline(
            np.percentile((grouped_dfs[i])['ERA5'], 95),
            color="k",
            linestyle="dashed",
            linewidth=1,
            label="ERA5 95th percentile",
        )

    plt.legend(loc="upper right")
    plt.show()


def mult_gauge_loc_plot(gauge_ds, other_datasets):
    """ Plot pdf of datasets for mulitple gauge locations."""

    gauge_df = gauge_ds.to_dataframe()
    ind_df = gauge_df.reset_index()
    melt_df = pd.melt(ind_df, id_vars=['time'], value_name='Gauge data')

    combined_df = melt_df[['time', 'Gauge data']]

    for ds in other_datasets:
        df1 = ds.to_dataframe()
        combined_df[ds.plot_legend] = df1.tp.values

    combined_df["time"] = np.ceil(
        (combined_df["time"] - np.floor(combined_df["time"])) * 12)
    combined_df["time"] = combined_df["time"].astype(int)

    month_list = []

    for m in range(1, 13):
        month_df = combined_df[combined_df['time'] == m]
        month_df = month_df.drop(['time'], axis=1)
        month_list.append(month_df)

    # Plot
    _fig, axs = plt.subplots(3, 4)  # , sharex=True, sharey=True)

    for i in range(12):
        x = (i) % 3
        y = int(i / 3)

        if i == 11:
            leg = True
        else:
            leg = False

        sns.kdeplot(data=month_list[i].drop(['Gauge data'], axis=1),
                    linestyle='--', ax=axs[x, y], bw_adjust=1, legend=leg,
                    common_norm=False)
        sns.kdeplot(data=month_list[i]['Gauge data'],
                    c='k', ax=axs[x, y], bw_adjust=1, legend=leg)

        axs[x, y].set_title(month_dict[i])
        axs[x, y].xaxis.set_tick_params(which="both", labelbottom=True)
        axs[x, y].yaxis.set_tick_params(which="both", labelbottom=True)
        axs[x, y].set_xlabel("Precipitation (mm/day)")
        axs[x, y].set_ylabel("Probability density")

    plt.show()
