""" 
Functions to help dowload data given
- Basin name
- Sub-basin name
- Coordinates
"""



def select_basin(dataset, location):
    """ Interpolate dataset at given coordinates """  
    mask_filepath = dp.find_mask(location)
    basin = dd.apply_mask(dataset, mask_filepath) 
    return basin


def find_mask(location):
    """ Returns a mask filepath for given location """

    mask_dic = {'ngari':'Data/Masks/Ngari_mask.nc', 'khyber':'Data/Masks/Khyber_mask.nc', 
                'gilgit':'Data/Masks/Gilgit_mask.nc', 'uib':'Data/Masks/ERA5_Upper_Indus_mask.nc',
                'sutlej': 'Data/Masks/Sutlej_mask.nc', 'beas':'Data/Masks/Beas_mask.nc'}
    mask_filepath = mask_dic[location]
    return mask_filepath