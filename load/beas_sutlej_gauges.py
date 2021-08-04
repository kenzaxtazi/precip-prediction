"""
Raw gauge measurements from the Beas and Sutlej valleys

"""
import numpy as np
import pandas as pd


all_station_dict = {'Arki':[31.154, 76.964], 'Banjar': [31.65, 77.34], 'Banjar IMD': [31.637, 77.344],  
                'Berthin':[31.471, 76.622], 'Bhakra':[31.424, 76.417], 'Barantargh': [31.087, 76.608], 
                'Bharmaur': [32.45, 76.533], 'Bhoranj':[31.648, 76.698], 'Bhuntar': [31.88, 77.15], 
                'Churah': [32.833, 76.167], 'Dadahu':[30.599, 77.437], 'Daslehra': [31.4, 76.55], 
                'Dehra': [31.885, 76.218], 'Dhaula Kuan': [30.517, 77.479], 'Ganguwal': [31.25, 76.486], 
                'Ghanauli': [30.994, 76.527], 'Ghumarwin': [31.436, 76.708], 'Hamirpur': [31.684, 76.519], 
                'Janjehl': [31.52, 77.22], 'Jogindernagar': [32.036, 76.734], 'Jubbal':[31.12, 77.67], 
                'Kalatop': [32.552, 76.018], 'Kalpa': [31.54, 78.258], 'Kandaghat': [30.965, 77.119], 
                'Kangra': [32.103, 76.271], 'Karsog': [31.383, 77.2], 'Kasol': [31.357, 76.878], 
                'Kaza': [32.225, 78.072], 'Kotata': [31.233, 76.534], 'Kothai': [31.119, 77.485],
                'Kumarsain': [31.317, 77.45], 'Larji': [31.80, 77.19], 'Lohard': [31.204, 76.561], 
                'Mashobra': [31.13, 77.229], 'Nadaun': [31.783, 76.35], 'Nahan': [30.559, 77.289], 
                'Naina Devi': [31.279, 76.554], 'Nangal': [31.368, 76.404], 'Olinda': [31.401, 76.385],
                'Pachhad': [30.777, 77.164], 'Palampur': [32.107, 76.543], 'Pandoh':[31.67,77.06], 
                'Paonta Sahib': [30.47, 77.625], 'Rakuna': [30.605, 77.473], 'Rampur': [31.454,77.644],
                'Rampur IMD': [31.452, 77.633], 'Rohru':[31.204, 77.751], 'Sadar-Bilarspur':[31.348, 76.762], 
                'Sadar-Mandi': [31.712, 76.933], 'Sainj': [31.77, 77.31] , 'Salooni':[32.728, 76.034],
                'Sarkaghat': [31.704, 76.812], 'Sujanpur':[31.832, 76.503], 'Sundernargar': [31.534, 76.905], 
                'Suni':[31.238,77.108], 'Suni IMD':[31.23, 77.164], 'Swaghat': [31.713, 76.746], 
                'Theog': [31.124, 77.347]}


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
    lat, lon = all_station_dict[station]
    df_ind = df.set_index('Date')
    ds = df_ind.to_xarray()
    ds = ds.assign_attrs(plot_legend="Gauge data")
    ds = ds.assign_coords({'lon':lon})
    ds = ds.assign_coords({'lat':lat})
    ds = ds.rename({'Date': 'time'})
    
    # Standardise time resolution
    raw_maxyear = float(ds.time.max())
    raw_minyear = float(ds.time.min())
    time_arr = np.arange(round(raw_minyear) + 1./24., raw_maxyear, 1./12.)
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

