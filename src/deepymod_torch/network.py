import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    '''Linear layer which also forwards the input data.'''
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
    def forward(self, input):
        if len(input) != 2: # for first layer
            X = input
            data = input
        else:
            X, data = input
        z = F.linear(X, self.weight, self.bias)
        return z, data


class Tanh(nn.Module):
    ''' Activation function which also forwards input data'''
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        X, data = input
        z = torch.tanh(X)
        return z, data


class Library(nn.Module):
    '''Abstract baseclass for library-as-layer. Child requires theta function (see library_functions). '''
    def __init__(self, library_func, library_args={}):
        super().__init__()
        self.library_func = library_func
        self.library_args = library_args

    def forward(self, input):
        '''Calculates output.'''
        time_deriv_list, theta = self.library_func(input, self.library_args)
        return (input, time_deriv_list, theta)


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
