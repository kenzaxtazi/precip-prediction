<p align="center" width="100%">
    <img width="30%" src="figures/logo_cropped.png">
</p>

# precip-prediction

Application of Gaussian Processes to predicting local precipitation over the Upper Indus Basin (HKKH).

## analysis

Functions to analyse data including:

- clustering
- correlation
- principal component analysis
- probability distribution functions
- timeseries

## gp

Functions to train and evaluate GPs using GPflow for precicting precipitation from other ERA5 climatic variables.

## load

Functions to load data and format data. The source of the different datasets used can be found in the submodule files. Please get in touch for gauge data, basin masks and shapefiles.

## maps

Functions to plot maps of the data and model outputs over the study areas.
