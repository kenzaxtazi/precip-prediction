"""
The APHRODITE dataset is available on in the BAS workspace on JASMIN
"""

def collect_APHRO():
    """ Downloads data from APHRODITE model"""
    aphro_ds = xr.merge([xr.open_dataset(f) for f in glob.glob('/Users/kenzatazi/Downloads/APHRO_MA_025deg_V1101.1951-2007.gz/*')])
    aphro_ds = aphro_ds.assign_attrs(plot_legend="APHRODITE") # in mm/day   
    aphro_ds = aphro_ds.rename_vars({'precip': 'tp'})
    aphro_ds['time'] = standardised_time(aphro_ds)
    return aphro_ds