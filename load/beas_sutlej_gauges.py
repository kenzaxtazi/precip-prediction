"""
Raw gauge measurements from the Beas and Sutlej valleys

"""
import numpy as np
import pandas as pd
from math import floor


all_station_dict = {'Arki':[31.154, 76.964, 1176], 'Banjar': [31.65, 77.34, 1914], 'Banjar IMD': [31.637, 77.344, 1427],  
                'Berthin':[31.471, 76.622, 657], 'Bhakra':[31.424, 76.417, 518], 'Barantargh': [31.087, 76.608, 285], 
                'Bharmaur': [32.45, 76.533, 1867], 'Bhoranj':[31.648, 76.698, 834], 'Bhuntar': [31.88, 77.15,834, 1100], 
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


def gauge_download(station, minyear, maxyear):
    """ 
    Download and format raw gauge data 
    
    Args:
        station (str): station name (capitalised)
    
    Returns
       df (pd.DataFrame): precipitation gauge values
    """
    filepath= 'Data/RawGauge_BeasSutlej_.xlsx'
    daily_df= pd.read_excel(filepath, sheet_name=station)
    # daily_df.dropna(inplace=True)

    # To months
    daily_df.set_index('Date', inplace=True)
    daily_df.index = pd.to_datetime(daily_df.index)
    clean_df = daily_df.dropna()
    df_monthly = clean_df.resample('M').sum()/30.436875
    df = df_monthly.reset_index()
    df['Date'] = df['Date'].values.astype(float)/365/24/60/60/1e9 +1970

    # to xarray DataSet
    lat, lon, _elv = all_station_dict[station]
    df_ind = df.set_index('Date')
    ds = df_ind.to_xarray()
    ds = ds.assign_attrs(plot_legend="Gauge data")
    ds = ds.assign_coords({'lon':lon})
    ds = ds.assign_coords({'lat':lat})
    ds = ds.rename({'Date': 'time'})
    
    # Standardise time resolution
    raw_maxyear = float(ds.time.max())
    print('max year', raw_maxyear)
    raw_minyear = float(ds.time.min())
    print('min year', raw_minyear)
    time_arr = np.arange(round(raw_minyear*12)/12 + 1./24., round(raw_maxyear*12)/12, 1./12.)
    print(time_arr[0:5], time_arr[-1])
    ds['time'] = time_arr
    
    tims_ds = ds.sel(time=slice(minyear, maxyear))
    return tims_ds


def all_gauge_data(minyear, maxyear, threshold=None):  #
    """
    Download data between specified dates for all active stations between two dates
    Can specify the treshold for the the total number of active days during that period
    E.g. for 10 year period -> 4018 - 365 = 3653
    """

    filepath = "Data/qc_sushiwat_observations_MGM.xlsx"
    daily_df= pd.read_excel(filepath)

    maxy_str = str(maxyear) + '-01-01'
    miny_str = str(minyear) + '-01-01'
    mask = (daily_df['Date'] >= miny_str ) & (daily_df['Date'] < maxy_str)
    df_masked = daily_df[mask]

    df_masked.set_index('Date', inplace=True)
    for col in list(df_masked):
        df_masked[col] = pd.to_numeric(df_masked[col], errors='coerce')
    
    if threshold != None:
        df_masked = df_masked.dropna(axis=1, thresh=threshold)

    df_masked.index = pd.to_datetime(df_masked.index)
    df_monthly = df_masked.resample('M').sum()/30.436875
    df = df_monthly.reset_index()
    df['Date'] = df['Date'].values.astype(float)/365/24/60/60/1e9 +1970

    df_ind = df.set_index('Date')
    ds = df_ind.to_xarray()
    ds = ds.assign_attrs(plot_legend="Gauge data")
    ds = ds.rename({'Date': 'time'})

    # Standardise time resolution
    time_arr = np.arange(round(minyear) + 1./24., maxyear, 1./12.)
    ds['time'] = time_arr

    return ds

