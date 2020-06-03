import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import PrecipitationDataExploration as pde
import FileDownloader as fd
import Clustering as cl
import GPModels as gpm

def R2(model,x_test, y_test):
    R2=1
    return R2