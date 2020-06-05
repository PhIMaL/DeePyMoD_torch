import torch
from deepymod_torch import Constraint


class LeastSquares(Constraint):
    ''' Calculates coefficients of constraint via least squares
        using QR decomposition. 
    '''
    def __init__(self):
        super().__init__()

    def calculate_coeffs(self, sparse_thetas, time_derivs):
        opt_coeff = []
        for theta, dt in zip(sparse_thetas, time_derivs):
            Q, R = torch.qr(theta)  # solution of lst. sq. by QR decomp.
            opt_coeff.append(torch.inverse(R) @ Q.T @ dt)
        return opt_coeff
