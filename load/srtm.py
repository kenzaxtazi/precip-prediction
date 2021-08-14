"""
SRTM dataset is a 90m DEM computed from satellite. 

Slope and aspect are calculated using:
    Horn, B.K.P., 1981. Hill shading and the reflectance map. Proceedings of the IEEE 69, 14–47. 
    doi:10.1109/PROC.1981.11918
"""

import xarray as xr
import richdem as rd
import numpy as np


def generate_slope_aspect():
    
    dem_ds = xr.open_dataset('/Users/kenzatazi/Downloads/GMTED2010_15n015_00625deg.nc')
    dem_ds = dem_ds.assign_coords({'nlat': dem_ds.latitude, 'nlon': dem_ds.longitude})
    dem_ds = dem_ds.sel(nlat=slice(29,34), nlon=slice(75, 83))

    elev_arr = dem_ds.elevation.values
    elev_rd_arr = rd.rdarray(elev_arr, no_data=np.nan)

    slope_rd_arr = rd.TerrainAttribute(elev_rd_arr, attrib='slope_riserun')
    slope_arr = np.array(slope_rd_arr)

    aspect_rd_arr = rd.TerrainAttribute(elev_rd_arr, attrib='aspect')
    aspect_arr = np.array(aspect_rd_arr)

    dem_ds['slope'] = (('nlat', 'nlon'), slope_arr)
    dem_ds['aspect'] = (('nlat', 'nlon'), aspect_arr)

    streamlined_dem_ds = dem_ds[['elevation', 'slope', 'aspect']]
    streamlined_dem_ds.to_netcdf('Data/SRTM_data.nc')


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


def find_slope(station):
    """ Returns slope for given station """
    dem_ds = xr.open_dataset('Data/SRTM_data.nc')
    location = all_station_dict[station]
    station_slope = dem_ds.interp(coords={"nlon": location[1], "nlat": location[0]}, method="nearest")
    return station_slope








