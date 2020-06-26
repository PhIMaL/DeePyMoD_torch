''' This file contains the building blocks for the deepmod framework. These are all abstract
    classes and implement the flow logic, rather than the specifics.
'''

import torch.nn as nn
import torch


class DeepMoD(nn.Module):
    '''Implements the deepmod model and connects all the separate elements.'''
    def __init__(self, function_approximator, library, sparsity_estimator, constraint):
        super().__init__()
        self.func_approx = function_approximator
        self.library = library
        self.sparse_estimator = sparsity_estimator
        self.constraint = constraint

    def forward(self, input):
        prediction = self.func_approx(input)
        time_derivs, theta = self.library((prediction, input))
        sparse_thetas, constraint_coeffs = self.constraint((time_derivs, theta))
        return prediction, time_derivs, sparse_thetas, theta, constraint_coeffs


class Library(nn.Module):
    ''' Library layer which calculates and normalizes the library.'''
    def __init__(self):
        super().__init__()
        self.norm = None

    def forward(self, input):
        time_derivs, theta = self.library(input)
        self.norm = torch.norm(theta, dim=0, keepdim=True)
        normed_theta = theta / self.norm  # we pass on the normed thetas
        return time_derivs, normed_theta


class Constraint(nn.Module):
    ''' Constraint layer which applies sparsity mask and calculates
        the coefficients.
    '''
    def __init__(self):
        super().__init__()
        self.sparsity_masks = None

    def forward(self, input):
        time_derivs, theta = input

        if self.sparsity_masks is None:
            self.sparsity_masks = [torch.ones(theta.shape[1], dtype=torch.bool) for _ in torch.arange(len(time_derivs))]

        sparse_thetas = self.apply_mask(theta)
        self.coeff_vectors = self.calculate_coeffs(sparse_thetas, time_derivs)
        return sparse_thetas, self.coeff_vectors

    def apply_mask(self, theta):
        sparse_thetas = [theta[:, sparsity_mask] for sparsity_mask in self.sparsity_masks]
        return sparse_thetas


class Estimator(nn.Module):
    ''' Estimator layer which implements calculation
        of sparsity mask.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, theta, time_derivs):
        self.coeff_vectors = [self.fit(theta.detach().cpu(), time_deriv.squeeze().detach().cpu())
                              for time_deriv in time_derivs]
        sparsity_masks = [torch.tensor(coeff_vector != 0.0, dtype=torch.bool)
                          for coeff_vector in self.coeff_vectors]

        return sparsity_masks
