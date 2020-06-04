import torch.nn as nn
import torch


class DeepMoD(nn.Module):
    def __init__(self, function_approximator, library, sparsity_estimator, constraint):
        super().__init__()
        self.func_approx = function_approximator
        self.library = library
        self.sparse_estimator = sparsity_estimator
        self.constraint = constraint

    def forward(self, input):
        prediction = self.func_approx(input)
        time_derivs, thetas = self.library((prediction, input))
        sparse_thetas, constraint_coeffs = self.constraint((time_derivs, thetas))
        return prediction, time_derivs, sparse_thetas, thetas, constraint_coeffs


class Library(nn.Module):
    ''' Library layer which calculates and normalizes the library.'''
    def __init__(self):
        super().__init__()
        self.norms = None

    def forward(self, input):
        time_derivs, thetas = self.library(input)
        self.norms = [torch.norm(theta, dim=0, keepdim=True) for theta in thetas]
        normed_thetas = [theta / norm for theta, norm in zip(thetas, self.norms)]  # we pass on the normed thetas
        return time_derivs, normed_thetas


class Constraint(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparsity_masks = None

    def forward(self, input):
        time_derivs, thetas = input
        sparse_thetas = self.apply_mask(thetas)
        self.coeff_vectors = self.calculate_coeffs(sparse_thetas, time_derivs)
        return sparse_thetas, self.coeff_vectors

    def apply_mask(self, thetas):
        if self.sparsity_masks is None:
            self.sparsity_masks = [torch.ones(theta.shape[1], dtype=torch.bool) for theta in thetas]
        sparse_theta = [theta[:, sparsity_mask] for theta, sparsity_mask in zip(thetas, self.sparsity_masks)]
        return sparse_theta


class Estimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, thetas, time_derivs):
        self.coeff_vectors = [self.fit(theta.detach().cpu(), time_deriv.squeeze().detach().cpu()) for theta, time_deriv in zip(thetas, time_derivs)]
        sparsity_masks = [torch.tensor(coeff_vector != 0.0, dtype=torch.bool) for coeff_vector in self.coeff_vectors]
        
        return sparsity_masks
