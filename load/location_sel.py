"""
Functions to help dowload data given
- Basin name
- Sub-basin name
- Coordinates
"""
import xarray as xr


def select_basin(dataset, location):
    """ Interpolate dataset at given coordinates """
    mask_filepath = find_mask(location)
    basin = apply_mask(dataset, mask_filepath)
    return basin


def find_mask(location):
    """ Returns a mask filepath for given location """
    mask_dic = {'ngari': '_Data/Masks/Ngari_mask.nc',
                'khyber': '_Data/Masks/Khyber_mask.nc',
                'gilgit': '_Data/Masks/Gilgit_mask.nc',
                'uib': '_Data/Masks/ERA5_Upper_Indus_mask.nc',
                'sutlej': '_Data/Masks/Sutlej_mask.nc',
                'beas': '_Data/Masks/Beas_mask.nc',
                'beas_sutlej': '_Data/Masks/Beas_Sutlej_mask.nc'}
    mask_filepath = mask_dic[location]
    return mask_filepath


def basin_finder(loc):
    """
    Finds basin to load data from.
    Input
        loc: list of coordinates [lat, lon] or string refering to an area.
    Output
        basin , string: name of the basin.
    """
    basin_dic = {'indus': 'indus', 'uib': 'indus', 'sutlej': 'indus',
                 'beas': 'indus', 'beas_sutlej': 'indus', 'khyber': 'indus',
                 'ngari': 'indus', 'gilgit': 'indus'}
    if type(loc) is str:
        basin = basin_dic[loc]
        return basin
    if type(loc) is not str:  # fix to search with coords
        print('Not a string')


def apply_mask(data, mask_filepath):
    """
    Opens NetCDF files and applies mask to data can also interp data to mask
    grid.
    Inputs:
        data (filepath string or NetCDF)
        mask_filepath (filepath string)
        interp (boolean): nearest-neighbour interpolation
    Return:
        A Data Array
    """
    if data is str:
        da = xr.open_dataset(data)
        if "expver" in list(da.dims):
            print("expver found")
            da = da.sel(expver=1)
    else:
        da = data
    mask = xr.open_dataset(mask_filepath)
    mask = mask.rename({'latitude': 'lat', 'longitude': 'lon'})
    mask_da = mask.overlap

    try:
        masked_da = da.where(mask_da > 0, drop=True)
    except Exception:
        interp_da = da.interp_like(mask_da)
        masked_da = interp_da.where(mask_da > 0, drop=True)
    return masked_da


def basin_extent(string):
    """ Returns extent of basin to save data """
    basin_dic = {'indus': [40, 65, 25, 85]}
    return basin_dic[string]
