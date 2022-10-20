<p align="center" width="100%">
    <img width="30%" src="figures/logo_cropped.png">
</p>

# precip-prediction

Application of Gaussian Processes to predicting local precipitation over the Upper Indus Basin (Himalayas).

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

## mfdgp

Functions to train and evalute Multi-Fidelity Deep Gaussian Processes using GPy and emukit.

## notebook1 (not finished)

Introduction to Himalayan precipitation data. This notebook introduces two precipiation datasets over the Upper Beas and Sutlej river basins, Himalayas. The aim is to give you a sense of differences between the two datasets (climate reanalysis and in-situ observations) but also a flavour of the complex spatio-temporal distribution of precpitation in this area.

## notebook2

Building a simple Multi-Fidelity Deep Gaussian Process. This notebook builds and trains a simple Multi-Fidelity Deep Gaussian Process (MFDGP) using the data presented in noteboook1.
