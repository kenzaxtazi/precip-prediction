# Feature selection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import feature_selection

import GPModels as gpm
import DataPreparation as dp

# Random sampling multivariate GP preparation
xtrain, xval, xtest, ytrain, yval, ytest = dp.random_multivariate_data_prep()


# Create the Pipeline and fit the feature selector
class PipelineRFE(Pipeline):
    
    def fit(self, X, y=None, **fit_params):
        super(PipelineRFE, self).fit(X, y, **fit_params)
        self.feature_importances_ = self.steps[-1][-1].feature_importances_
        return self


pipe = PipelineRFE([("RF", gpm.multi_gp(xtrain, xval, ytrain, yval))])

feature_selector_cv = feature_selection.RFECV(pipe, step=1, scoring="neg_mean_squared_error", verbose=3)
feature_selector_cv.fit(xtrain, ytrain)

# Plot the RMSE as a function of the number of features
cv_grid_rmse = np.sqrt(-feature_selector_cv.grid_scores_)

plt.plot(cv_grid_rmse)
plt.title('RMSE versus number of features')
plt.show()

# Based on the analysing the graph and computational considerations, decide on a number of features
selected_features = [f for f, r in zip(X.columns, feature_selector_cv.ranking_) if r < number_of_features]
print(selected_features)
