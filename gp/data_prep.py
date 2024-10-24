# Data Preparation

import sys
import numpy as np
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# custom libraries
dir_path = '/Users/kenzatazi/Documents/CDT/Code/' # noqa
# dir_path = '/data/hpcdata/users/kenzi22/' # noqa
sys.path.append(dir_path)  # noqa

from load import era5, location_sel
import gp.sampling as sa


class point_model():

    def __init__(self, location: str | np.ndarray, seed=42, all_data=True, all_var=False):
        """
        Output training, validation and test sets for total precipitation.

        Args:
            location (str | np.ndarray): 'uib', 'khyber', 'ngari', 'gilgit' or [lat, lon] coordinates.
            number (int, optional): ensemble run number. Defaults to None.
            EDA_average (bool, optional): use ERA5 low-res ensemble average. Defaults to False.
            maxyear (str, optional): end year (inclusive). Defaults to None.
            seed (int, optional): sampling random generator seed. Defaults to 42.
            all_var (bool, optional): all the variables studied if True or only final selection for paper if False. Defaults to False.

        Returns:
            tuple: contains
                x_train (np.array): training feature vector
                x_val (np.array): validation feature vector
                x_test (np.array): testing feature vector
                y_train (np.array): training output vector
                y_val (np.array): validation output vector
                y_test (np.array): testing output vector
                lmbda: lambda value for Box Cox transformation
        """

        # Download data 
        if all_data is True:
            df_train_all = pd.read_csv(dir_path + 'precip-prediction/data/uib_train_all.csv')
        if all_data is False:
            df_train_all = pd.read_csv(dir_path + 'precip-prediction/data/uib_train_7000_'+ str(seed)+'.csv')
        
        df_val_all = pd.read_csv(dir_path + 'precip-prediction/data/uib_val_1000_'+ str(seed)+'.csv')
        df_test_all = pd.read_csv(dir_path + 'precip-prediction/data/uib_test_2000_'+ str(seed)+'.csv')

        # Find location
        df_train_loc = df_train_all[(df_train_all['lon'] == location[0]) & (df_train_all['lat'] == location[1])]
        df_val_loc = df_val_all[(df_val_all['lon'] == location[0]) & (df_val_all['lat'] == location[1])]
        df_test_loc = df_test_all[(df_test_all['lon'] == location[0]) & (df_test_all['lat'] == location[1])] 

        # Variable choice
        if all_var is True:
            var_list = ["time", "tcwv", "d2m", "EOF200U",  "t2m", "EOF850U",  "EOF500U", "EOF500B2", "EOF200B",
                "NAO", "EOF500U2", "N34", "EOF850U2", "EOF500B", "tp",]
        else:
            var_list =["time", "tcwv", "EOF200U", "EOF500U", "tp"] #"d2m", "t2m", "tp",]

        df_train = df_train_loc[var_list]
        df_val = df_val_loc[var_list]
        df_test = df_test_loc[var_list]

        # Standardize time
        df_train["time"] = pd.to_datetime(df_train["time"])
        df_train["time"] = pd.to_numeric(df_train["time"])
        df_val["time"] = pd.to_datetime(df_val["time"])
        df_val["time"] = pd.to_numeric(df_val["time"])
        df_test["time"] = pd.to_datetime(df_test["time"])
        df_test["time"] = pd.to_numeric(df_test["time"])

        df_train['tp'].loc[df_train['tp'] <= 0.0] = 0.0001
        df_val['tp'].loc[df_val['tp'] <= 0.0] = 0.0001
        df_test['tp'].loc[df_test['tp'] <= 0.0] = 0.0001
        
        # Keep first of 70% for training
        xtrain = df_train.drop(columns=["tp"]).values
        ytrain = df_train['tp'].values
        xval = df_val.drop(columns=["tp"]).values
        yval = df_val['tp'].values
        xtest = df_test.drop(columns=["tp"]).values
        ytest = df_test['tp'].values

        # Precipitation transformation
        ytrain_tr, lmbda = sp.stats.boxcox(ytrain)
        yval_tr = sp.stats.boxcox(yval, lmbda=lmbda)
        ytest_tr = sp.stats.boxcox(ytest, lmbda=lmbda)
   
        # Features scaling
        xscaler = MinMaxScaler()
        xtrain = xscaler.fit_transform(xtrain.astype(np.float64))
        xval = xscaler.transform(xval)
        xtest = xscaler.transform(xtest)

        yscaler = StandardScaler()
        ytrain_sc = yscaler.fit_transform(ytrain_tr.reshape(-1,1))
        yval_sc = yscaler.transform(yval_tr.reshape(-1,1))
        ytest_sc = yscaler.transform(ytest_tr.reshape(-1,1))

        # Set class variables    
        self.ytrain = ytrain
        self.yval = yval
        self.ytest = ytest

        self.ytrain_tr = ytrain_tr
        self.yval_tr = yval_tr
        self.ytest_tr = ytest_tr

        self.ytrain_sc = ytrain_sc
        self.yval_sc = yval_sc
        self.ytest_sc = ytest_sc

        self.xtrain = xtrain
        self.xtest = xtest
        self.xval = xval

        self.l = lmbda
        self.xscaler = xscaler
        self.yscaler = yscaler

    def sets(self):
        return self.xtrain, self.xval, self.xtest, self.ytrain_sc, self.yval_sc, self.ytest_sc



class areal_model_new():
    """ Class for generating data for areal models"""

    def __init__(self, location, seed=42, var='uib'):
        """
        Inputs
            location: specify area to train model
            number, optional: specify desired ensemble run, integer
            EDA_average, optional: specify if you want average of low resolution
                ensemble runs, boolean
            length, optional: specify number of points to sample for training, integer
            seed, optional: specify seed, integer

        Outputs
            x_train: training feature vector, numpy array
            y_train: training output vector, numpy array
            x_test: testing feature vector, numpy array
            y_test: testing output vector, numpy array
        """

        # Download data 
        df_train_all = pd.read_csv(dir_path + 'precip-prediction/data/uib_train_7000_'+ str(seed)+'.csv')
        df_val_all = pd.read_csv(dir_path + 'precip-prediction/data/uib_val_1000_'+ str(seed)+'.csv')
        df_test_all = pd.read_csv(dir_path + 'precip-prediction/data/uib_test_2000_'+ str(seed)+'.csv')

        train_ds = df_train_all.set_index(['lat', 'lon', 'time']).to_xarray()
        val_ds = df_val_all.set_index(['lat', 'lon', 'time']).to_xarray()
        test_ds = df_test_all.set_index(['lat', 'lon', 'time']).to_xarray()

        # Apply mask
        mask_filepath = location_sel.find_mask(location)
        train_masked_ds = location_sel.apply_mask(train_ds, mask_filepath)
        val_masked_ds = location_sel.apply_mask(val_ds, mask_filepath)
        test_masked_ds = location_sel.apply_mask(test_ds, mask_filepath)


        # Choose variables
        if var == "all":
            var_list = ["time", "lon", "lat", "tcwv", "slor", "d2m", "z", "EOF200U", "t2m", "EOF850U",  "EOF500U", "EOF500B2", "EOF200B",
                        "anor", "NAO", "EOF500U2", "N34", "EOF850U2", "EOF500B2", "EOF500C" ,"EOF500C2", "tp"]
        elif var == "uib":
            var_list = ["time", "lon", "lat", "slor", "d2m", "z", "EOF200U", "t2m", "EOF850U",  "EOF500U", "tp"]
        elif var == "khyber":
            var_list = ["time", "lon", "lat", "slor", "d2m", "z", "EOF200U", "t2m", "NAO", "tp"]
        elif var == "ngari": 
            var_list = ["time", "lon", "lat", "slor", "d2m", "EOF200U", "t2m", "EOF850U", "NAO", "tp"]
        elif var == "gilgit":
            var_list = ["time", "lon", "lat", "slor", "EOF200U", "t2m", "EOF850U", "EOF500B2", "anor", "NAO", "tp"]

        df_train = train_masked_ds.to_dataframe().reset_index().dropna() 
        df_val = val_masked_ds.to_dataframe().reset_index().dropna()
        df_test = test_masked_ds.to_dataframe().reset_index().dropna()

        df_train = df_train[var_list]
        df_val = df_val[var_list]
        df_test = df_test[var_list]

        ### Training and evaluation dataset

        #### Training data

        df_train["time"] = pd.to_datetime(df_train["time"])
        df_train["time"] = pd.to_numeric(df_train["time"])
        df_val["time"] = pd.to_datetime(df_val["time"])
        df_val["time"] = pd.to_numeric(df_val["time"])
        df_test["time"] = pd.to_datetime(df_test["time"])
        df_test["time"] = pd.to_numeric(df_test["time"])

    
        # Get rid of zeros
        df_train['tp'].loc[df_train['tp'] <= 0.0] = 0.0001
        df_val['tp'].loc[df_val['tp'] <= 0.0] = 0.0001
        df_test['tp'].loc[df_test['tp'] <= 0.0] = 0.0001

        # X and Y
        ytrain = df_train['tp'].values
        xtrain = df_train.drop(columns=["tp"]).values
        yval = df_val['tp'].values
        xval = df_val.drop(columns=["tp"]).values
        ytest = df_test['tp'].values
        xtest = df_test.drop(columns=["tp"]).values

        # Transformation
        ytrain_tr, l = sp.stats.boxcox(ytrain.astype(np.float64))
        yval_tr = sp.stats.boxcox(yval, lmbda=l)
        ytest_tr = sp.stats.boxcox(ytest, lmbda=l)

        # Features scaling
        xscaler = MinMaxScaler()
        xtrain = xscaler.fit_transform(xtrain.astype(np.float64))
        xval = xscaler.transform(xval)
        xtest = xscaler.transform(xtest)

        yscaler = StandardScaler()
        ytrain_sc = yscaler.fit_transform(ytrain_tr.reshape(-1,1))
        yval_sc = yscaler.transform(yval_tr.reshape(-1,1))
        ytest_sc = yscaler.transform(ytest_tr.reshape(-1,1))

        # Set class variables    
        self.ytrain = ytrain
        self.yval = yval
        self.ytest = ytest

        self.ytrain_tr = ytrain_tr
        self.yval_tr = yval_tr
        self.ytest_tr = ytest_tr

        self.ytrain_sc = ytrain_sc
        self.yval_sc = yval_sc
        self.ytest_sc = ytest_sc

        self.xtrain = xtrain
        self.xtest = xtest
        self.xval = xval

        self.l = l
        self.xscaler = xscaler
        self.yscaler = yscaler

    def sets(self):
        return self.xtrain, self.xval, self.xtest, self.ytrain_sc, self.yval_sc, self.ytest_sc
    
'''

class point_model():

    def __init__(self, location: str | np.ndarray, number:int=None, EDA_average:bool=False, maxyear:str=None, seed=42, all_var=False):
        """
        Output training, validation and test sets for total precipitation.

        Args:
            location (str | np.ndarray): 'uib', 'khyber', 'ngari', 'gilgit' or [lat, lon] coordinates.
            number (int, optional): ensemble run number. Defaults to None.
            EDA_average (bool, optional): use ERA5 low-res ensemble average. Defaults to False.
            maxyear (str, optional): end year (inclusive). Defaults to None.
            seed (int, optional): sampling random generator seed. Defaults to 42.
            all_var (bool, optional): all the variables studied if True or only final selection for paper if False. Defaults to False.

        Returns:
            tuple: contains
                x_train (np.array): training feature vector
                x_val (np.array): validation feature vector
                x_test (np.array): testing feature vector
                y_train (np.array): training output vector
                y_val (np.array): validation output vector
                y_test (np.array): testing output vector
                lmbda: lambda value for Box Cox transformation
        """
        # End year for data
        if maxyear is None:
            maxyear = '2020'

        # Variable list
        if all_var is True:
            var_list = ["time", "tcwv", "d2m", "EOF200U",  "t2m", "EOF850U",  "EOF500U", "EOF500B2", "EOF200B",
                "NAO", "EOF500U2", "N34", "EOF850U2", "EOF500B", "tp",]
        else:
            var_list =["time", "tcwv", "EOF200U", "EOF500U", "tp"] #"d2m", "t2m", "tp",]
        
        # Download data and format data for string location
        if isinstance(location, str) == True:
            if number is not None:
                da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
                da = da_ensemble.sel(number=number).drop("number")
            if EDA_average is True:
                da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
                da = da_ensemble.mean(dim="number")
            else:
                da = era5.download_data(location, xarray=True, all_var=True,)

            multiindex_df = da.to_dataframe()
            df_clean = multiindex_df.dropna().reset_index()
            df_location = sa.random_location_sampler(df_clean)
            df = df_location.drop(columns=["lat", "lon", "slor", "anor", "z"])

        # Download data and format data for coordinates
        if isinstance(location, np.ndarray) == True:
            da_location = era5.collect_ERA5(
                (location[0], location[1]), minyear='1970', maxyear=maxyear, all_var=True)
            multiindex_df = da_location.to_dataframe()
            df_clean = multiindex_df.dropna().reset_index()
            df = df_clean.drop(columns=["lat", "lon", "slor", "anor", "z"])

        # Standardize time and select variables
        df["time"] = pd.to_datetime(df["time"])
        df["time"] = pd.to_numeric(df["time"])
        df = df[var_list]
        
        # Keep first of 70% for training
        x = df.drop(columns=["tp"]).values
        df.loc[df['tp'] <= 0.0] = 0.0001
        y = df['tp'].values

        xtrain, x_eval, ytrain, y_eval = train_test_split(
            x, y, test_size=0.3, shuffle=False) 

        # Last 30% for evaluation
        xval, xtest, yval, ytest = train_test_split(x_eval, 
            y_eval, test_size=24, train_size=12, shuffle=True, random_state=seed)

        # Precipitation transformation
        ytrain_tr, lmbda = sp.stats.boxcox(ytrain)
        yval_tr = sp.stats.boxcox(yval, lmbda=lmbda)
        ytest_tr = sp.stats.boxcox(ytest, lmbda=lmbda)
   
        # Features scaling
        xscaler = StandardScaler()
        xtrain = xscaler.fit_transform(xtrain.astype(np.float64))
        xval = xscaler.transform(xval)
        xtest = xscaler.transform(xtest)

        yscaler = StandardScaler()
        ytrain_sc = yscaler.fit_transform(ytrain_tr.reshape(-1,1))
        yval_sc = yscaler.transform(yval_tr.reshape(-1,1))
        ytest_sc = yscaler.transform(ytest_tr.reshape(-1,1))

        # Set class variables    
        self.ytrain = ytrain
        self.yval = yval
        self.ytest = ytest

        self.ytrain_tr = ytrain_tr
        self.yval_tr = yval_tr
        self.ytest_tr = ytest_tr

        self.ytrain_sc = ytrain_sc
        self.yval_sc = yval_sc
        self.ytest_sc = ytest_sc

        self.xtrain = xtrain
        self.xtest = xtest
        self.xval = xval

        self.l = lmbda
        self.xscaler = xscaler
        self.yscaler = yscaler

    def sets(self):
        return self.xtrain, self.xval, self.xtest, self.ytrain_sc, self.yval_sc, self.ytest_sc



class areal_model_new():
    """ Class for generating data for areal models"""

    def __init__(self, location, number=None, EDA_average=False, length=3000, seed=42,
                maxyear=None, minyear='1970', var=False):
        """
        Inputs
            location: specify area to train model
            number, optional: specify desired ensemble run, integer
            EDA_average, optional: specify if you want average of low resolution
                ensemble runs, boolean
            length, optional: specify number of points to sample for training, integer
            seed, optional: specify seed, integer

        Outputs
            x_train: training feature vector, numpy array
            y_train: training output vector, numpy array
            x_test: testing feature vector, numpy array
            y_test: testing output vector, numpy array
        """
        if maxyear is None:
            maxyear = '2020'

        if number is not None:
            da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
            da = da_ensemble.sel(number=number).drop("number")

        if EDA_average is True:
            da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
            da = da_ensemble.mean(dim="number")
        else:
            ds = era5.collect_ERA5(location, minyear="1970", maxyear=maxyear, all_var=True)

        # Apply mask
        mask_filepath = location_sel.find_mask(location)
        masked_ds = location_sel.apply_mask(ds, mask_filepath)

        if var == "all":
            var_list = ["time", "lon", "lat", "tcwv", "slor", "d2m", "z", "EOF200U", "t2m", "EOF850U",  "EOF500U", "EOF500B2", "EOF200B",
                        "anor", "NAO", "EOF500U2", "N34", "EOF850U2", "EOF500B2", "EOF500C" ,"EOF500C2", "tp"]
        elif var == "uib":
            var_list = ["time", "lon", "lat", "slor", "d2m", "z", "EOF200U", "t2m", "EOF850U",  "EOF500U", "tp"]
        elif var == "khyber":
            var_list = ["time", "lon", "lat", "slor", "d2m", "z", "EOF200U", "t2m", "NAO", "tp"]
        elif var == "ngari": 
            var_list = ["time", "lon", "lat", "slor", "d2m", "EOF200U", "t2m", "EOF850U", "NAO", "tp"]
        elif var == "gilgit":
            var_list = ["time", "lon", "lat", "slor", "EOF200U", "t2m", "EOF850U", "EOF500B2", "anor", "NAO", "tp"]           

        ### Training and evaluation dataset

        train_ds = masked_ds.sel(time=slice('1970', '2005'))
        eval_ds = masked_ds.sel(time=slice('2005', '2020'))

        #### Training data
        multiindex_df = train_ds.to_dataframe()
        df_train = multiindex_df.dropna().reset_index()
        df_train["time"] = pd.to_datetime(df_train["time"])
        df_train["time"] = pd.to_numeric(df_train["time"])
        df_train = df_train[var_list]
        df_train['tp'].loc[df_train['tp'] <= 0.0] = 0.0001

        # Sample training
        df_train_samp = sa.random_location_and_time_sampler(df_train, by_loc=False, length=length, seed=123)  
        xtrain = df_train_samp.drop(columns=["tp"]).values
        ytrain = df_train_samp['tp'].values

        #### Evaluation data
        multiindex_df = eval_ds.to_dataframe() 
        df_eval = multiindex_df.dropna().reset_index()
        df_eval["time"] = pd.to_datetime(df_eval["time"])
        df_eval["time"] = pd.to_numeric(df_eval["time"])

        df_eval = df_eval[var_list]
        #df_eval[df_eval['tp'] <= 0.0] = 0.0001
        
        loc_df = df_eval.groupby(['lat','lon']).mean().reset_index()
        locs = loc_df[['lon','lat']].values

        xval_list = []
        yval_list = []
        xtest_list = []
        ytest_list = []  

        for i in range(len(locs)):
            lon, lat = locs[i]
            loc_df0 = df_eval[(df_eval['lat'] == lat) & (df_eval['lon'] == lon)]
            loc_df = loc_df0.dropna()
            loc_df['tp'].loc[loc_df['tp'] <= 0.0] = 0.0001
            # sample 20% of the location
            #loc_df = loc_df.sample(frac=0.1, random_state=seed)
            xeval = loc_df.drop(columns=["tp"]).values
            yeval = loc_df['tp'].copy(deep=False).values
            loc_xval, loc_xtest, loc_yval, loc_ytest = train_test_split(xeval, yeval, test_size=24, 
                                                                        train_size=12, shuffle=True, 
                                                                        random_state=seed)
            xval_list.extend(loc_xval)
            yval_list.extend(loc_yval)
            xtest_list.extend(loc_xtest)
            ytest_list.extend(loc_ytest)
        
        xval = np.array(xval_list, dtype=np.float64)
        yval = np.array(yval_list, dtype=np.float64)
        xtest = np.array(xtest_list, dtype=np.float64)
        ytest = np.array(ytest_list, dtype=np.float64)

        # Training and validation data
        
        # Transformations
        ytrain_tr, l = sp.stats.boxcox(ytrain.astype(np.float64))
        yval_tr = sp.stats.boxcox(yval, lmbda=l)
        ytest_tr = sp.stats.boxcox(ytest, lmbda=l)

        # Features scaling
        xscaler = StandardScaler()
        xtrain = xscaler.fit_transform(xtrain.astype(np.float64))
        xval = xscaler.transform(xval)
        xtest = xscaler.transform(xtest)

        yscaler = StandardScaler()
        ytrain_sc = yscaler.fit_transform(ytrain_tr.reshape(-1,1))
        yval_sc = yscaler.transform(yval_tr.reshape(-1,1))
        ytest_sc = yscaler.transform(ytest_tr.reshape(-1,1))

        # Set class variables    
        self.ytrain = ytrain
        self.yval = yval
        self.ytest = ytest

        self.ytrain_tr = ytrain_tr
        self.yval_tr = yval_tr
        self.ytest_tr = ytest_tr

        self.ytrain_sc = ytrain_sc
        self.yval_sc = yval_sc
        self.ytest_sc = ytest_sc

        self.xtrain = xtrain
        self.xtest = xtest
        self.xval = xval

        self.l = l
        self.xscaler = xscaler
        self.yscaler = yscaler

    def sets(self):
        return self.xtrain, self.xval, self.xtest, self.ytrain_sc, self.yval_sc, self.ytest_sc

def areal_model(location:str, number:int=None, EDA_average=False, length=3000, seed=42,
                maxyear=None):
    """
    Outputs test, validation and training data for total precipitation as a
    function of time, 2m dewpoint temperature, angle of sub-gridscale
    orography, orography, slope of sub-gridscale orography, total column water
    vapour, Nino 3.4 index for given number randomly sampled data points
    for a given basin.

    Inputs
        location: specify area to train model
        number, optional: specify desired ensemble run, integer
        EDA_average, optional: specify if you want average of low resolution
            ensemble runs, boolean
        length, optional: specify number of points to sample, integer
        seed, optional: specify seed, integer

    Outputs
        x_train: training feature vector, numpy array
        y_train: training output vector, numpy array
        x_test: testing feature vector, numpy array
        y_test: testing output vector, numpy array
    """

    if number is not None:
        da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop("number")

    if EDA_average is True:
        da_ensemble = era5.download_data(location, xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = era5.download_data(location, xarray=True, all_var=True)

    # apply mask
    mask_filepath = location_sel.find_mask(location)
    masked_da = location_sel.apply_mask(da, mask_filepath)

    if maxyear is not None:
        df["time"] = df[df["time"] < maxyear]

    multiindex_df = masked_da.to_dataframe()
    df_clean = multiindex_df.dropna().reset_index()
    df = sa.random_location_and_time_sampler(
        df_clean, length=length, seed=seed)

    df["time"] = pd.to_datetime(df["time"])
    df["time"] = pd.to_numeric(df["time"])
    # df["tp"] = log_transform(df["tp"])
    df = df[["tcwv", "slor", "d2m", "lon", "z", "EOF200U", "t2m", "EOF850U", "tp"]]  # format order

    # Keep first of 70% for training
    x = df.drop(columns=["tp"]).values
    df[df['tp'] <= 0.0] = 0.0001

    y = df['tp'].values

    # Last 30% for evaluation
    xtrain, x_eval, ytrain, y_eval = train_test_split(
        x, y, test_size=0.3, shuffle=False,)

    # Training and validation data
    xval, xtest, yval, ytest = train_test_split(x_eval, 
        y_eval, test_size=1./3., shuffle=True, random_state=seed)

    # Transformations
    ytrain_tr, l = sp.stats.boxcox(ytrain)
    yval_tr = sp.stats.boxcox(yval, lmbda=l)
    ytest_tr = sp.stats.boxcox(ytest, lmbda=l)

    return xtrain, xval, xtest, ytrain_tr, yval_tr, ytest_tr, l



def areal_model_eval(location, lmbda, number=None, EDA_average=False, length=3000,
                     seed=42, minyear="1970", maxyear="2020"):
    """
    Returns data to evaluate an areal model at a given location, area and time
    period.

    Variables:
        Total precipitation as a function of time, 2m dewpoint
        temperature, angle of sub-gridscale orography, orography, slope of
        sub-gridscale orography, total column water vapour, Nino 3.4, Nino 4
        and NAO index for a single point.

    Inputs:
        number, optional: specify desired ensemble run, integer
        EDA_average, optional: specify if you want average of low resolution
            ensemble runs, boolean
        coords [latitude, longitude], optional: specify if you want a specific
            location, list of floats
        mask, optional: specify area to train model, defaults to Upper Indus
            Basin

    Outputs
        x_tr: evaluation feature vector, numpy array
        y_tr: evaluation output vector, numpy array
    """
    if number is not None:
        da_ensemble = era5.download_data('uib', xarray=True, ensemble=True)
        da = da_ensemble.sel(number=number).drop("number")
    if EDA_average is True:
        da_ensemble = era5.download_data('uib', xarray=True, ensemble=True)
        da = da_ensemble.mean(dim="number")
    else:
        da = era5.download_data('uib', xarray=True)

    sliced_da = da.sel(time=slice(minyear, maxyear))

    if isinstance(location, str) is True:
        mask_filepath = location_sel.find_mask(location)
        masked_da = location_sel.apply_mask(sliced_da, mask_filepath)
        multiindex_df = masked_da.to_dataframe()
        multiindex_df = da.to_dataframe()
        df_clean = multiindex_df.dropna().reset_index()
        df = sa.random_location_and_time_sampler(
            df_clean, length=length, seed=seed)

    else:
        da_location = sliced_da.interp(
            coords={"lat": location[0], "lon": location[1]}, method="nearest"
        )
        multiindex_df = da_location.to_dataframe()
        df = multiindex_df.dropna().reset_index()

    df["time"] = pd.to_numeric(pd.to_datetime(df["time"])) # to years
    df["tp_tr"] = sp.stats.boxcox(df["tp"].values, lmbda=lmbda)
    df = df[["time", "lat", "lon", "d2m", "tcwv", "N34", "tp_tr"]]  # format order
    #df = df[["time", "lat", "lon", "slor", "anor", "z", "d2m", "tcwv", "N34", "tp_tr"]]  # format order
    
    xtr = df.drop(columns=["tp_tr"]).values
    ytr = df["tp_tr"].values

    return xtr, ytr


def average_over_coords(ds):
    """Take average over latitude and longitude"""
    ds = ds.mean("lon")
    ds = ds.mean("lat")
    return ds


def shared_evaluation_sets(location:str, length=3000, seed=42):

    uib_ds = era5.collect_ERA5('uib', minyear="2005", maxyear="2020")
    df_clean = uib_ds.to_dataframe().dropna()
    df = sa.random_location_and_time_sampler(df_clean, length=length, seed=seed)
    df_eval = df[["tcwv", "slor", "d2m", "lon", "z", "EOF200U", "t2m", "EOF850U", "tp"]]

    y_df = df_eval["tp"]
    x_df = df_eval.drop(columns=["tp"])
    xval_uib_df, xtest_uib_df, yval_uib_df, ytest_uib_df = train_test_split(x_df, y_df, test_size=1./3., shuffle=True, random_state=seed)

    if location == 'uib':
        xval_df = xval_uib_df.reset_index()
        xval_df['time'] = pd.to_datetime(xval_df['time'])
        xval_df['time'] = pd.to_numeric(xval_df['time'])
        xval = xval_df.values

        xtest_df = xtest_uib_df.reset_index()
        xtest_df['time'] = pd.to_datetime(xtest_df['time'])
        xtest_df['time'] = pd.to_numeric(xtest_df['time'])
        xtest = xtest_df.values

        yval, ytest =  yval_uib_df.values, ytest_uib_df.values

    if location in ('khyber', 'ngari', 'gilgit'):
        mask_filepath = location_sel.find_mask(location)

        xval_uib_df = xval_uib_df[~xval_uib_df.index.duplicated()]
        xtest_uib_df = xtest_uib_df[~xtest_uib_df.index.duplicated()]
        yval_uib_df = yval_uib_df[~yval_uib_df.index.duplicated()]
        ytest_uib_df = ytest_uib_df[~ytest_uib_df.index.duplicated()]
        
        xval_uib_ds = xval_uib_df.to_xarray()
        xtest_uib_ds = xtest_uib_df.to_xarray()
        yval_uib_ds = yval_uib_df.to_xarray()
        ytest_uib_ds = ytest_uib_df.to_xarray()

        xval_ds = location_sel.apply_mask(xval_uib_ds, mask_filepath)
        xtest_ds = location_sel.apply_mask(xtest_uib_ds, mask_filepath)
        yval_ds = location_sel.apply_mask(yval_uib_ds, mask_filepath)
        ytest_ds = location_sel.apply_mask(ytest_uib_ds, mask_filepath)

        xval_df = xval_ds.to_dataframe().reset_index().dropna()
        xtest_df = xtest_ds.to_dataframe().reset_index().dropna()
        yval_df = yval_ds.to_dataframe().dropna()
        ytest_df = ytest_ds.to_dataframe().dropna()

        xval_df['time'] = pd.to_datetime(xval_df['time'])
        xval_df['time'] = pd.to_numeric(xval_df['time'])
        xval = xval_df.values
        
        xtest_df['time'] = pd.to_datetime(xtest_df['time'])
        xtest_df['time'] = pd.to_numeric(xtest_df['time'])
        xtest = xtest_df.values

        yval, ytest =  yval_uib_df.values, ytest_uib_df.values

    if location is isinstance(tuple):
        xtest_uib_df = xtest_uib_df.reset_index()
        ytest_uib_df = ytest_uib_df.reset_index()
        xval_uib_df = xval_uib_df.reset_index()
        yval_uib_df = yval_uib_df.reset_index()

        xval_df = xval_uib_df[(xval_uib_df['lat']== location[0]) & (xval_uib_df['lon'] == location[1])]
        yval_df = yval_uib_df[(yval_uib_df['lat']== location[0]) & (yval_uib_df['lon'] == location[1])]
        xtest_df = xtest_uib_df[(xtest_uib_df['lat']== location[0]) & (xtest_uib_df['lon'] == location[1])]
        ytest_df = ytest_uib_df[(ytest_uib_df['lat']== location[0]) & (ytest_uib_df['lon'] == location[1])]

        xval = xval_df[["time", "tcwv", "d2m", "tp"]]
        yval = yval_df["tp"]
        xtest = xtest_df[["time", "tcwv", "d2m", "tp"]]
        ytest = ytest_df["tp"]

    return  xval, xtest, yval, ytest

'''