import torch
import torch.nn as nn


class LstSq(nn.Module):
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

    def calculate_coeffs(self, sparse_thetas, time_derivs):
        opt_coeff = []
        for theta, dt in zip(sparse_thetas, time_derivs):
            Q, R = torch.qr(theta)  # solution of lst. sq. by QR decomp.
            opt_coeff.append(torch.inverse(R) @ Q.T @ dt)
        return opt_coeff
