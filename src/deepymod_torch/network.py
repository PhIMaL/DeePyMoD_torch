import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    '''Pytorch style linear layer which also calculates the derivatives w.r.t input. Has been written to be a thin wrapper around the pytorch layer. '''
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
    def forward(self, input):
        '''Calculates output'''
        X, dX = input
        z = F.linear(X, self.weight, self.bias)
        dz = F.linear(dX, self.weight)
        return (z, dz)


class ActivationFunction(nn.Module):
    '''Abstract baseclass for activation function layer.  '''
    def __init__(self):
        super().__init__()
        self.layer_func_derivs = [lambda f, g: g[:, 0, :, :] * f[:, 1:2, :],
                         lambda f, g: g[:, 0, :, :]**2 * f[:, 2:3, :] + g[:, 1, :, :] * f[:, 1:2, :],
                         lambda f, g: 3 * g[:, 0, :, :] * g[:, 1, :, :] * f[:, 2:3, :] + g[:, 0, :, :]**3 * f[:, 3:4, :] + g[:, 2, :, :] * f[:, 1:2, :]] #ordered list of derivatives of activation layer
        
    def forward(self, input):
        '''Calculates output'''
        dsigma = []
        for order in range(input[1].shape[1]+1):
            dsigma.append(self.activation_func_derivs[order](dsigma, input[0])) # Calculating pure derivatives of the activation function
        dsigma = torch.stack(dsigma, dim=1)
      
        df = torch.stack([self.layer_func_derivs[order](dsigma, input[1]) for order in range(input[1].shape[1])], dim=1) # calculating total derivative of activation function
        f = dsigma[:, 0, :]
    
        return (f, df)


class Tanh(ActivationFunction):
    '''Tanh activation layer baseclass.'''
    def __init__(self):
        super().__init__()
        self.activation_func_derivs = [lambda ds, x: torch.tanh(x),
                                       lambda ds, x: 1 / torch.cosh(x)**2,
                                       lambda ds, x: -2 * ds[0] * ds[1],
                                       lambda ds, x: ds[2]**2 / ds[1] - 2* ds[1]**2] # ordered list of activation function and its derivatives


class Library(nn.Module):
    '''Abstract baseclass for library-as-layer. Child requires theta function (see library_functions). '''
    def __init__(self, n_in, n_out, library_func, library_args={}):
        super().__init__()
        self.library_func = library_func
        self.library_args = library_args
        self.n_terms = self.terms(n_in, n_out)
        
    def forward(self, input):
        '''Calculates output.'''
        time_deriv_list, theta = self.library_func(input, **self.library_args)
        return (input, time_deriv_list, theta)

    def terms(self, n_in, n_out):
        '''Calculates the number of terms the library produces'''
        max_order = self.library_args['diff_order']
        sample_data = (torch.ones((1, n_out), dtype=torch.float32), torch.ones((1, max_order, n_in, n_out), dtype=torch.float32)) # we run a single forward pass on fake data to infer shapes
        total_terms = self.forward(sample_data)[2].shape[1]

        return total_terms

class Fitting(nn.Module):
    def __init__(self, n_terms, n_out):
        super().__init__()
        self.coeff_vector = nn.ParameterList([torch.nn.Parameter(torch.rand((n_terms, 1), dtype=torch.float32)) for _ in torch.arange(n_out)])
        self.sparsity_mask = [torch.arange(n_terms) for _ in torch.arange(n_out)]

    def forward(self, input):
        prediction, time_deriv, theta = input
        sparse_theta = self.apply_mask(theta)
        return prediction, time_deriv, sparse_theta, self.coeff_vector

    def apply_mask(self, theta):
        sparse_theta = [theta[:, sparsity_mask] for sparsity_mask in self.sparsity_mask]
        return sparse_theta
