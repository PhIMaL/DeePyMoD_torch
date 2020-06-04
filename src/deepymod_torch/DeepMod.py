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

    def calculate_sparsity_mask(self, thetas, time_derivs):
        coeff_vectors = [self.sparse_estimator.fit(theta.detach().cpu(), time_deriv.squeeze().detach().cpu()).coef_ for theta, time_deriv in zip(thetas, time_derivs)]
        sparsity_masks = [torch.tensor(coeff_vector != 0.0, dtype=torch.bool) for coeff_vector in coeff_vectors]
        return sparsity_masks
