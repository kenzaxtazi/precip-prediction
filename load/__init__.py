"""
Datasets should all have the same format so they can be easily used together.

In particular, they should be exported from the submodule:
- as a xarray DataArray or saved asnetcdf file format
- with 'lon' as the longitude variable name in °E (float)
- with 'lat' as the latitude variable name in °N (float)
- with 'time' for time variable name in years with monthly resolution taken in
the middle of each month (float)
- with 'tp' for the variable name for total precipitation in mm/day (float)
"""
