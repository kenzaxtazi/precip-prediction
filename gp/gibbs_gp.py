import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/precip-prediction/')

import gpytorch 
import torch
import numpy as np
from gpytorch.kernels import InducingPointKernel,  ScaleKernel, PeriodicKernel, MaternKernel
from gp.multivariate_gibbs_kernel import MultivariateGibbsKernel

gpytorch.settings.cholesky_jitter(1e-4)

## Declaring model class -- its easier to have this in the script as one can experiment with different settings - for instance, fixing or training inducing locations

class MultiGibbsKernel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, Z_init, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        self.base_covar_module = ScaleKernel(MultivariateGibbsKernel(Z_init, 2))
        self.spatial_covar_module = InducingPointKernel(self.base_covar_module, inducing_points=Z_init, likelihood=likelihood)
        self.spatial_covar_module.inducing_points.requires_grad_(False)
        #self.spatial_covar_module = ScaleKernel(MaternKernel(nu=1.5, active_dims=(1,2)))

        self.temporal_covar_module = ScaleKernel(PeriodicKernel(active_dims=[0]) * MaternKernel(nu=1.5, active_dims=[0]))
        self.clim_covar_module = ScaleKernel(MaternKernel(nu=1.5, active_dims=[3]))
        for i in range(4, train_x.shape[1]):
             self.clim_covar_module += ScaleKernel(MaternKernel(nu=1.5, active_dims=[i]))

        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.temporal_covar_module(x) + self.spatial_covar_module(x[:,(1,2)]) + self.clim_covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
