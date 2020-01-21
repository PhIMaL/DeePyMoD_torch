import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Library(nn.Module):
    def __init__(self, input_dim, output_dim, diff_order):
        super().__init__()
        self.diff_order = diff_order
        self.total_terms = self.terms(input_dim, output_dim, self.diff_order)
        self.sparsity_mask_list = [torch.arange(self.total_terms) for _ in torch.arange(output_dim)]
        self.coeff_vector_list = nn.ParameterList([torch.nn.Parameter(torch.rand((self.total_terms, 1), dtype=torch.float32)) for _ in torch.arange(output_dim)])

    def forward(self, input):
        dt, theta = self.theta(input)
        sparse_theta_list = [theta[:, sparsity_mask] for sparsity_mask in self.sparsity_mask_list]
        return input, dt, sparse_theta_list, self.coeff_vector_list

    def terms(self, input_dim, output_dim, max_order):
        sample_data = (torch.ones((1, output_dim), dtype=torch.float32), torch.ones((1, max_order, input_dim, output_dim), dtype=torch.float32)) # we run a single forward pass on fake data to infer shapes
        total_terms = self.theta(sample_data)[1].shape[1]

        return total_terms