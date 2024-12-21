#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multivariate Gibbs Kernel -> Matrix GP prior with Paciorek and Scheverish form.

"""
import gpytorch 
import torch
import math
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel, InducingPointKernel
from gpytorch.likelihoods import GaussianLikelihood
import numpy as np

global jitter
jitter = 1e-5
softplus = torch.nn.Softplus()

class IndependentMatrixPrior(gpytorch.priors.MultivariateNormalPrior):
    
   ''' 
   Independent GP Priors for rows of  D^{2} x N matrix 
   
   '''
   def __init__(self, X):
       
       mean_module = gpytorch.means.ZeroMean()
       covar_module = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=2))
       covar_module.outputscale = 0.25
       covar_module.base_kernel.lengthscale = [1.0, 3.0]
       
       covar_matrix = (covar_module(X).evaluate() + jitter*torch.eye(X.shape[0])).detach()
       mean_vector = mean_module(X)
       
       super().__init__(loc=mean_vector.double(), covariance_matrix=covar_matrix.double())
       
       self.n = X.shape[0]
       self.d = X.shape[1]
       self.k = self.d**2
       self.covar_module = covar_module
       self.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
       self.covar_module.raw_outputscale.requires_grad_(False)
       #self.covar_matrix = self.covar_matrix.detach()
       

   def sample_h(self):
       H = super().sample((self.k,))
       return H
   
   def inverse(self):
       return torch.linalg.inv(self.covar_matrix)

class MultivariateGibbsKernel(gpytorch.kernels.Kernel):
    
    """
    Multivariate Gibbs kernel -> GP prior combined with Paciorek and Scheverish [2003] form.
    """
    is_stationary = False

    # We will register the parameter when initializing the kernel
    def __init__(self, x, input_dim, **kwargs):
        super().__init__(**kwargs)
        
        self.x = x
        self.n = len(x)
        self.d = input_dim
        
        #if 'Z' in kwargs:
        #    self.Z = kwargs['Z']
        
        if input_dim == 1: 
            raise ValueError('Use gibbs 1d kernel for dim 1')
        
        else: 
            
            self.H_matrix_prior = IndependentMatrixPrior(self.x) 
            H_init = self.H_matrix_prior.sample_h()
            self.register_parameter(name='H', parameter=torch.nn.Parameter(H_init.to(torch.float64)))
            self.register_prior('prior_H', self.H_matrix_prior, 'H')
            
            #D_init = torch.diag(torch.randn(2))
            #self.register_parameter(name='D', parameter=torch.nn.Parameter(D_init.to(torch.float32)))
            
    def expectation_conditional_matrix_variate_dist(self, x_star):
        
        K_h_inv =  self.prior_H.covariance_matrix.inverse()    # N x N / M x M
        K_star_h = self.prior_H.covar_module(x_star, self.x).evaluate() # N* x N        
        pre_prod = torch.matmul(K_star_h, K_h_inv)
        cond_mean = torch.Tensor(np.array([torch.matmul(pre_prod, h.T).detach().numpy() for h in self.H]))
        return cond_mean
                
    def forward(self, x1, x2, diag=False, **params):
                
        if torch.equal(x1, x2):
            
            try: 
                
                assert(len(x1) == self.H.shape[1])
            
                Hx = self.H.detach()
                N1 = N2 = len(x1)
                
            except AssertionError: 
                
                # this means the size of the H matrix and size of the inputs don't match but the two 
                # inputs are the same -> kernel on test inputs K_{**} is being computed.
                
                Hx = self.expectation_conditional_matrix_variate_dist(x1).detach()
                N1 = N2 = len(x1)
                
            # x1 is NxD
            
            raw_sigmas = torch.Tensor(np.array([x.reshape(self.d, self.d).numpy() for x in Hx.T])) # N x D x D or N* x D x D.
            sigma_xs = torch.Tensor(np.array([np.array(torch.matmul(x, x.T) + 1e-6*torch.eye(self.d)) for x in raw_sigmas]))
            
            ## expanding to get dimensions NxN
            self.sigma_matrix_i = torch.einsum('ijkl->jikl', sigma_xs.expand(N1,N1,self.d,self.d)) # each row has a DxD matrix
            self.sigma_matrix_j = torch.einsum('ijkl->jikl', self.sigma_matrix_i)
            
            sigma_dets_matrix = torch.det(self.sigma_matrix_i).pow(0.25) 
            det_product = torch.mul(sigma_dets_matrix, sigma_dets_matrix.T) ## N x N
        
        else:
            
            ## x1 and x2 are different data blocks -- computing the test-train cross covariance
            #print('Detected x1 and x2 are different - need cross covariance computation')
                    
            if x1.shape[0] == self.H.shape[1]: ## x1 is training as its length matches the column length of H
            
                x_star = x2
                Hx1 = self.H.detach() 
                Hx2 = self.expectation_conditional_matrix_variate_dist(x_star).detach()
                
            elif x2.shape[0] == self.H.shape[1]:
                
                x_star = x1
                Hx2 = self.H.detach()
                Hx1 = self.expectation_conditional_matrix_variate_dist(x_star).detach()
                            
            N1 = len(x1)
            N2 = len(x2)
            
            raw_sigmas1 = torch.Tensor(np.array([x.reshape(self.d, self.d).numpy() for x in Hx1.T]))
            sigma_x1 = torch.Tensor(np.array([np.array(torch.matmul(x, x.T) + 1e-5*torch.eye(self.d)) for x in raw_sigmas1])) # N1 x D x D
            
            raw_sigmas2 = torch.Tensor(np.array([x.reshape(self.d, self.d).numpy() for x in Hx2.T]))
            sigma_x2 = torch.Tensor(np.array([np.array(torch.matmul(x, x.T) + 1e-5*torch.eye(self.d)) for x in raw_sigmas2])) # N2 x D x D
        
            ## expanding to get dimensions NxN
            
            sigma_matrix_i = torch.einsum('ijkl->jikl', sigma_x1.expand(N1,N1,self.d,self.d)) # each row has a DxD matrix
            self.sigma_matrix_i = sigma_matrix_i[:, 0:N2, :,:]
            self.sigma_matrix_j = sigma_x2.expand(N1,N2,self.d, self.d)
            
            sigma_dets_matrix_i = torch.det(self.sigma_matrix_i).pow(0.25) 
            sigma_dets_matrix_j = torch.det(self.sigma_matrix_j).pow(0.25) 
            det_product = torch.mul(sigma_dets_matrix_i, sigma_dets_matrix_j) ## N1 x N2
            
        avg_kernel_matrix = (self.sigma_matrix_i + self.sigma_matrix_j)/2
        avg_kernel_det = torch.det(avg_kernel_matrix).pow(-0.5) 
        self.prefactor = torch.mul(det_product, avg_kernel_det) ## N1 x N2
        
        self.sig_inv = torch.inverse(avg_kernel_matrix + jitter*torch.eye(self.d)).double() ## N1 x N2 x D x D
        self.diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)) 
        self.first_prod = torch.matmul(self.diff.unsqueeze(-2), self.sig_inv)
        self.final_prod = torch.matmul(self.first_prod, self.diff.unsqueeze(-1)).reshape(N1,N2) ## N1xN2

        if torch.equal(x1, x2):
            
            return torch.mul(self.prefactor, torch.exp(-self.final_prod)) + 1e-4*torch.eye(len(x1)) ## N1 x N2
        
        else:
            
            return torch.mul(self.prefactor, torch.exp(-self.final_prod))