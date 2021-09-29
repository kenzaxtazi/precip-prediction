import FileDownloader as fd


def indice_downloader(all_var=False):
    """ Return indice Dataframe """

    nao_url = "https://www.psl.noaa.gov/data/correlation/nao.data"
    n34_url = "https://psl.noaa.gov/data/correlation/nina34.data"
    n4_url = "https://psl.noaa.gov/data/correlation/nina4.data"

    n34_df = fd.update_url_data(n34_url, "N34")

    if all_var is False:
        ind_df = n34_df.astype("float64")
    else:
        nao_df = fd.update_url_data(nao_url, "NAO")
        n4_df = fd.update_url_data(n4_url, "N4")
        ind_df = n34_df.join([nao_df, n4_df])

    return ind_df
