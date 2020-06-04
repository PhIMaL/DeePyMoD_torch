import torch.nn as nn


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
