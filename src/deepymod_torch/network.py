import torch
import torch.nn as nn


class Library(nn.Module):
    """
    A module that represents a library of functions used in the DeepMoD algorithm.

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
    # TODO: write a docstring
    def __init__(self, n_terms, n_out):
        super().__init__()
        self.coeff_vector = nn.ParameterList([torch.nn.Parameter(torch.rand((n_terms, 1), dtype=torch.float32)) for _ in torch.arange(n_out)])
        self.sparsity_mask = [torch.arange(n_terms) for _ in torch.arange(n_out)]

    def forward(self, input):
        sparse_theta = self.apply_mask(input)
        return sparse_theta, self.coeff_vector

    def apply_mask(self, theta):
        sparse_theta = [theta[:, sparsity_mask] for sparsity_mask in self.sparsity_mask]
        return sparse_theta
