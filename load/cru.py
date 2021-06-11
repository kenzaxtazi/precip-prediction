"""
CRU dataset
"""


def collect_CRU():
    """ Downloads data from CRU model"""
    cru_ds = xr.open_dataset("Data/cru_ts4.04.1901.2019.pre.dat.nc")
    cru_ds = cru_ds.assign_attrs(plot_legend="CRU") # in mm/month
    cru_ds = cru_ds.rename_vars({'pre': 'tp'})
    cru_ds['tp'] /= 30.437  #TODO apply proper function to get mm/day
    cru_ds['time'] = standardised_time(cru_ds)
    return cru_ds