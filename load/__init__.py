"""
Datasets should aim to all have similar formats so they can be easily used and compared.

- lon for longitude in °E (float)
- lat for latitude in °N (float)
- time for time in years (float) with monthly resolution taken in the middle of each month
- tp for total precipitation in mm/day/month (float)

Datasets should be returned as xarray Datasets and saved as netcdf files when possible.
"""


# Dataset class?
