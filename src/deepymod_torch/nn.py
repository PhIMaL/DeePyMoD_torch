import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


def create_deriv_data(X, max_order):
    ''' Utility function to create the deriv object'''
    if max_order == 1:
        dX = (torch.eye(X.shape[1]) * torch.ones(X.shape[0])[:, None, None])[:, None, :]
    else: 
        dX = [torch.eye(X.shape[1]) * torch.ones(X.shape[0])[:, None, None]]
        dX.extend([torch.zeros_like(dX[0]) for order in range(max_order-1)])
        dX = torch.stack(dX, dim=1)
        
    return (X, dX)

class Linear(nn.Linear):
    ''' Linear layer completely similar to Pytorch's; only takes in deriv data tuple and also outputs it.'''
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
    def forward(self, input):
        X, dX = input
        z = F.linear(X, self.weight, self.bias)
        dz = F.linear(dX, self.weight)
        return (z, dz)


class ActivationFunction(nn.Module):
    '''Abstract baseclass for activation functions which also outputs derivatives. '''
    def __init__(self):
        super().__init__()
        self.layer_func_derivs = [lambda f, g: g[:, 0, :, :] * f[:, 1:2, :],
                         lambda f, g: g[:, 0, :, :]**2 * f[:, 2:3, :] + g[:, 1, :, :] * f[:, 1:2, :],
                         lambda f, g: 3 * g[:, 0, :, :] * g[:, 1, :, :] * f[:, 2:3, :] + g[:, 0, :, :]**3 * f[:, 3:4, :] + g[:, 2, :, :] * f[:, 1:2, :]] #ordered list of derivatives of activation layer
        
    def forward(self, input):
        dsigma = []
        for order in range(input[1].shape[1]+1):
            dsigma.append(self.activation_func_derivs[order](dsigma, input[0])) # Calculating pure derivatives of the activation function
        dsigma = torch.stack(dsigma, dim=1)
      
        df = torch.stack([self.layer_func_derivs[order](dsigma, input[1]) for order in range(input[1].shape[1])], dim=1) # calculating total derivative of activation function
        f = dsigma[:, 0, :]
    
        return (f, df)


class Tanh(ActivationFunction):
    '''Tanh-specific activation function.'''
    def __init__(self):
        super().__init__()
        self.activation_func_derivs = [lambda ds, x: torch.tanh(x),
                                       lambda ds, x: 1 / torch.cosh(x)**2,
                                       lambda ds, x: -2 * ds[0] * ds[1],
                                       lambda ds, x: ds[2]**2 / ds[1] - 2* ds[1]**2] # ordered list of activation function and its derivatives
       