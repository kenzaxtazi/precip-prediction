import os
import urllib
import numpy as np
import datetime
import pandas as pd


def indice_downloader(all_var=False):
    """ Return indice Dataframe."""

    nao_url = "https://www.psl.noaa.gov/data/correlation/nao.data"
    n34_url = "https://psl.noaa.gov/data/correlation/nina34.data"
    n4_url = "https://psl.noaa.gov/data/correlation/nina4.data"

    n34_df = update_url_data(n34_url, "N34")

    if all_var is False:
        ind_df = n34_df.astype("float64")
    else:
        nao_df = update_url_data(nao_url, "NAO")
        n4_df = update_url_data(n4_url, "N4")
        ind_df = n34_df.join([nao_df, n4_df])

    return ind_df


def save_csv_from_url(url, saving_path):
    """Downloads data from a url and saves it to a specified path."""
    response = urllib.request.urlopen(url)
    with open(saving_path, "wb") as f:
        f.write(response.read())


def update_url_data(url, name):
    """
    Import the most recent dataset from URL and return it as pandas DataFrame.
    """

    filepath = "_Data/NOAA/"
    now = datetime.datetime.now()
    file = filepath + name + "-" + now.strftime("%m-%Y") + ".csv"

    # Only download CSV if not present locally
    if not os.path.exists(file):
        save_csv_from_url(url, file)

    # create and format DataFrame
    df = pd.read_csv(file)
    df_split = df[list(df)[0]].str.split(expand=True)
    df_long = pd.melt(
        df_split,
        id_vars=[0],
        value_vars=np.arange(1, 13),
        var_name="month",
        value_name=name,
    )

    # Create a datetime column
    df_long["time"] = df_long[0].astype(
        str) + "-" + df_long["month"].astype(str)
    df_long["time"] = pd.to_datetime(df_long["time"], errors="coerce")

    # Clean
    df_clean = df_long.dropna()
    df_sorted = df_clean.sort_values(by=["time"])
    df_final = df_sorted.set_index("time")

    return pd.DataFrame(df_final[name])
