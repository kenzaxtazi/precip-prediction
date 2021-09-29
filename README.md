# Precipitation prediction

Application of probabilistic machine learning methods to predicting local precipitation over the Upper Indus Basin (Himalayas).

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

Functions to load data and format data. The source of the different datasets used can be found in the submodule files. Please get in touch for the basin masks and shapefiles.

## maps

Functions to plot maps data and study areas.

## mfdgp

Functions to train and evalute Multi-Fidelity Deep Gaussian Processes using GPy and emukit.