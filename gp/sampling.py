# Sampling

import numpy as np
from load import era5

mask_filepath = "_Data/ERA5_Upper_Indus_mask.nc"


def random_location_sampler(df):
    """ Returns DataFrame of random location, apply to clean df only """

    df_squished = df[["lat", "lon"]].reset_index()
    df_s_reset = df_squished.drop_duplicates()
    i = np.random.randint(len(df_s_reset), size=1)

    df_location = df_s_reset.iloc[i]
    lat = df_location["lat"].values[0]
    lon = df_location["lon"].values[0]
    print("lat=" + str(lat) + ", lon=" + str(lon))

    df1 = df[df["lat"] == lat]
    df2 = df1[df1["lon"] == lon]

    return df2


def random_location_generator(location, N=50):
    """ Returns DataFrame of random location, apply to clean df only """

    coord_list = []

    df = era5.download_data(location)
    df_squished = df[["lat", "lon"]].reset_index()
    df_s_reset = df_squished.drop_duplicates(subset=["lat", "lon"])

    indices = np.random.randint(len(df_s_reset), size=N)

    for i in indices:
        df_location = df_s_reset.iloc[i]
        lat = df_location["lat"]
        lon = df_location["lon"]
        coord_list.append([lat, lon])

    return coord_list


def random_location_and_time_sampler(df, length=1000, by_loc=False, seed=42):
    """Return DataFrame of random locations and times."""

    np.random.seed(seed)
    df_sorted = df.sort_values(by='time')

    if by_loc==False:
        i = np.random.randint(len(df), size=length)
        df_sampled = df_sorted.iloc[i]

    if by_loc==True:
        J = np.random.randint(len(df)-36, size=int(length/36))
        for j in J:
            J = np.append(J, np.arange(j+1, j+36))
        print(J.shape)
        df_sampled = df_sorted.iloc[J]

    return df_sampled
