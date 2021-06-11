"""
Raw gauge measurements from the Beas and Sutlej valleys

"""

import pandas as pd


station_dict = {'Banjar': [31.65,77.34], 'Sainj': [31.77,76.933], 'Ganguwal': [31.25,76.486]}


def gauge_download(station, xarray=False):
    """ 
    Download and format raw gauge data 
    
    Args:
        station (str): station name (capitalised)
    
    Returns
       df (pd.DataFrame): precipitation gauge values
    """
    filepath= 'Data/RawGauge_BeasSutlej_.xlsx'
    daily_df= pd.read_excel(filepath, sheet_name=station)
    daily_df.dropna(inplace=True)

    # To months
    daily_df.set_index('Date', inplace=True)
    daily_df.index = pd.to_datetime(daily_df.index)
    clean_df = daily_df.dropna()
    df_monthly = clean_df.resample('M').sum()/30.436875
    df = df_monthly.reset_index()
    df['Date'] = df['Date'].values.astype(float)/365/24/60/60/1e9 +1970
    clean_df2 = df.dropna()

    if xarray == False:
        return clean_df2
    
    else:
        lat, lon = station_dict[station]
        df_ind = clean_df2.set_index('Date')
        ds = df_ind.to_xarray()
        ds = ds.assign_attrs(plot_legend="Gauge data")
        ds = ds.assign_coords({'lon':lon})
        ds = ds.assign_coords({'lat':lat})
        ds = ds.rename({'Date': 'time'})
        return ds
