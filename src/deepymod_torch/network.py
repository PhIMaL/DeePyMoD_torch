import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Linear):
    '''
    Pytorch style linear layer which also calculates the derivatives w.r.t input. Has been written to be a thin
    wrapper around the pytorch layer. 

    Attributes
    ----------
    in_features : int
        Number of neurons in input of layer.
    out_features : int
        Number of neurons in output of layer.
    bias : Boolean
        Whether to apply bias.
    '''
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        
    def forward(self, input):
        '''
        Forward function for linear layer. 

        Parameters
        ----------
        input : TensorTuple
            Data-Deriv tuple containing the data and the deriv object.

        Returns
        -------
        output : Data-Deriv tuple
            Data-Deriv tuple containing the data and the deriv object.
        '''
        X, dX = input
        z = F.linear(X, self.weight, self.bias)
        dz = F.linear(dX, self.weight)
        return (z, dz)


class ActivationFunction(nn.Module):
    '''
    Abstract baseclass for activation function layer. 

    Attributes
    ----------
    layer_func_derivs : list
        List of lambda functions used to calculate derivatives.
    '''
    def __init__(self):
        super().__init__()
        self.layer_func_derivs = [lambda f, g: g[:, 0, :, :] * f[:, 1:2, :],
                         lambda f, g: g[:, 0, :, :]**2 * f[:, 2:3, :] + g[:, 1, :, :] * f[:, 1:2, :],
                         lambda f, g: 3 * g[:, 0, :, :] * g[:, 1, :, :] * f[:, 2:3, :] + g[:, 0, :, :]**3 * f[:, 3:4, :] + g[:, 2, :, :] * f[:, 1:2, :]] #ordered list of derivatives of activation layer
        
    def forward(self, input):
        '''
        Forward function for linear layer. 

        Parameters
        ----------
        input : TensorTuple
            Data-Deriv tuple containing the data and the deriv object.

        Returns
        -------
        output : Data-Deriv tuple
            Data-Deriv tuple containing the data and the deriv object.
        '''
        dsigma = []
        for order in range(input[1].shape[1]+1):
            dsigma.append(self.activation_func_derivs[order](dsigma, input[0])) # Calculating pure derivatives of the activation function
        dsigma = torch.stack(dsigma, dim=1)
      
        df = torch.stack([self.layer_func_derivs[order](dsigma, input[1]) for order in range(input[1].shape[1])], dim=1) # calculating total derivative of activation function
        f = dsigma[:, 0, :]
    
        return (f, df)


class Tanh(ActivationFunction):
    '''
    Tanh activation layer baseclass

    Attributes
    ----------
    activation_func_derivs : List
        List of lambda functions used to calculate activaiton function derivatives.

    '''
    def __init__(self):
        super().__init__()
        self.activation_func_derivs = [lambda ds, x: torch.tanh(x),
                                       lambda ds, x: 1 / torch.cosh(x)**2,
                                       lambda ds, x: -2 * ds[0] * ds[1],
                                       lambda ds, x: ds[2]**2 / ds[1] - 2* ds[1]**2] # ordered list of activation function and its derivatives


class Library(nn.Module):
    '''
    Abstract baseclass for library-as-layer. Child requires theta function (see library_functions).

    Attributes
    ----------
    input_dim : int
        Number of neurons in input layer of NN.
    output_dim : int
        Number of neurons in output of NN.
    diff_order : int
        Higher order derivative
    '''
    def __init__(self, input_dim, output_dim, diff_order):
        super().__init__()
        self.diff_order = diff_order
        self.total_terms = self.terms(input_dim, output_dim, self.diff_order)
        self.sparsity_mask_list = [torch.arange(self.total_terms) for _ in torch.arange(output_dim)]
        self.coeff_vector_list = nn.ParameterList([torch.nn.Parameter(torch.rand((self.total_terms, 1), dtype=torch.float32)) for _ in torch.arange(output_dim)])

    def forward(self, input):
        '''
        Forward function for library layer. Calls library function and applies sparsity mask.

        Parameters
        ----------
        input : TensorTuple
            Data-Deriv tuple containing the data and the deriv object.

        Returns
        -------
        input : tensor list of length of M
            list containing the time derivatives, each entry corresponds to an equation.
        time_deriv_list : Tensorlist
            List of tensors containing time derivatives.
        sparse_theta_list : Tensorlist
            List of tensors containing sparse libraries.
        coeff_vector_list : ParameterList
            List of Parameters containing coefficient vectors.
        '''
        time_deriv_list, theta = self.theta(input)
        sparse_theta_list = [theta[:, sparsity_mask] for sparsity_mask in self.sparsity_mask_list] # Applies sparsity mask so we get sparse theta
        return input, time_deriv_list, sparse_theta_list, self.coeff_vector_list

    def terms(self, input_dim, output_dim, max_order):
        # Calculates the number of terms the library produces
        sample_data = (torch.ones((1, output_dim), dtype=torch.float32), torch.ones((1, max_order, input_dim, output_dim), dtype=torch.float32)) # we run a single forward pass on fake data to infer shapes
        total_terms = self.theta(sample_data)[1].shape[1]

        return total_terms