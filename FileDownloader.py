import calendar
import datetime
import os
import urllib
import numpy as np
import xarray as xr
import pandas as pd
import ftplib
import cdsapi


nao_url = 'https://www.psl.noaa.gov/data/correlation/nao.data'
n34_url =  'https://psl.noaa.gov/data/correlation/nina34.data'
n4_url =  'https://psl.noaa.gov/data/correlation/nina4.data'


def save_csv_from_url(url, saving_path):
	""" Downloads data from a url and saves it to a specified path. """
	response = urllib.request.urlopen(url)
	with open(saving_path, 'wb') as f:
		f.write(response.read())


def update_url_data(url, name):
    """ Import the most recent dataset from URL and return it as pandas DataFrame """
    
    filepath = '/Users/kenzatazi/Downloads/'
    now = datetime.datetime.now()
    file = filepath + name + '-' + now.strftime("%m-%Y") + '.csv'

    # Only download CSV if not present locally
    if not os.path.exists(file):
        save_csv_from_url(url, file)
    
    # create and format DataFrame 
    df = pd.read_csv(file)
    df_split = df[list(df)[0]].str.split(expand=True)
    df_long = pd.melt(df_split, id_vars=[0], value_vars=np.arange(1,13), var_name='month', value_name=name)
    
    # Create a datetime column
    df_long['time'] = df_long[0].astype(str) + '-' + df_long['month'].astype(str)
    df_long['time'] = pd.to_datetime(df_long['time'], errors='coerce') 
    
    # Clean
    df_clean = df_long.dropna()
    df_sorted = df_clean.sort_values(by=['time'])
    df_final = df_sorted.set_index('time')

    return pd.DataFrame(df_final[name])


def update_cds_data(dataset_name='reanalysis-era5-single-levels-monthly-means',
                    product_type= 'monthly_averaged_reanalysis',
                    variables = 'total_precipitation',
                    area = [40, 70, 30, 85],
                    path = '/Users/kenzatazi/Downloads/'):
    """
    Imports the most recent version of the given ERA5 dataset as a netcdf from the CDS API.
    
    Inputs:
        dataset_name: str 
        prduct_type: str
        variables: list of strings
        area: list of scalars
        path: str

    Returns: local filepath to netcdf.
    """

    now = datetime.datetime.now()
    filename = dataset_name + '_' + product_type + '_' + now.strftime("%m-%Y")+'.nc' # TODO include variables in pathname

    filepath = path + filename 

    # Only download if updated file is not present locally
    if not os.path.exists(filepath):
        
        current_year = now.strftime("%Y")
        years = np.arange(1979, int(current_year)+1, 1).astype(str)
        months = np.arange(1, 13, 1).astype(str)

        c = cdsapi.Client()
        c.retrieve('reanalysis-era5-single-levels-monthly-means',
                {'format': 'netcdf',
                    'product_type': 'monthly_averaged_reanalysis',
                    'variable': variables,
                    'year': years.tolist(),
                    'time': '00:00',
                    'month': months.tolist(),
                    'area': area},
                    filepath)
    
    return filepath


def update_to_dataframe():

    # indices
    nao_df = fd.update_url_data(nao_url, 'NAO')
    n34_df = fd.update_url_data(n34_url, 'N34')
    n4_df = fd.update_url_data(n4_url, 'N4')
    ind_df = nao_df.join([n34_df, n4_df]).astype('float64')


    # Orography, humidity and precipitation
    cds_filepath = fd.update_cds_data(variables=['2m_dewpoint_temperature', 'angle_of_sub_gridscale_orography', 
                                                'orography', 'slope_of_sub_gridscale_orography', 
                                                'total_column_water_vapour', 'total_precipitation'])
    masked_da = pde.apply_mask(cds_filepath, mask_filepath)
    gilgit = masked_da.interp(coords={'longitude':74.4584, 'latitude':35.8884 }, method='nearest')
    multiindex_df = gilgit.to_dataframe()
    cds_df = multiindex_df.reset_index()

    # Combine
    df_combined = pd.merge_ordered(cds_df, ind_df, on='time') 

    return

    