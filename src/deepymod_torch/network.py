import torch
import torch.nn as nn


class Library(nn.Module):
    """
    A module subclass that represents a library of functions used in the DeepMoD algorithm.

    Args:
        library_func (callable): A function that generates the library of functions.
        library_args (dict): A dictionary of arguments to be passed to the library function.

    Returns:
        tuple: A tuple containing the time derivative list and the theta matrix.
    """
    def __init__(self, library_func, library_args={}):
        super().__init__()
        self.library_func = library_func
        self.library_args = library_args

    def forward(self, input):
        time_deriv_list, theta = self.library_func(input, **self.library_args)
        return time_deriv_list, theta


class Fitting(nn.Module):
    """
    A submodule for attaching a sparse regression layer to the DeepMoD model.

    Args:
        n_terms (int): The number of terms in the linear combination.
        n_out (int): The number of output features.

    Attributes:
        coeff_vector (nn.ParameterList): A list of learnable coefficient vectors, one for each output feature.
        sparsity_mask (list): A list representing sparsity mask, length corresponds to n_out.

    Methods:
        forward(input): Computes the sparse linear combination of input features for each output feature.
        apply_mask(theta): Applies the sparsity mask to the input features.
    """
    def __init__(self, n_terms, n_out):
        super().__init__()
        self.coeff_vector = nn.ParameterList([torch.nn.Parameter(torch.rand((n_terms, 1), dtype=torch.float32)) for _ in torch.arange(n_out)])
        self.sparsity_mask = [torch.arange(n_terms) for _ in torch.arange(n_out)]
        # sparse_theta is theta with sparsity mask applied (extracting relevant terms)
        # coeff_vector will play role in the loss function (see `losses.py`) which explains how it is optimized
    def forward(self, input):
        sparse_theta = self.apply_mask(input)
        return sparse_theta, self.coeff_vector

    def apply_mask(self, theta):
        sparse_theta = [theta[:, sparsity_mask] for sparsity_mask in self.sparsity_mask]
        return sparse_theta
