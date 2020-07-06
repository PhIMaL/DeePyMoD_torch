import torch
from .deepmod import Constraint
from ..utils.types import TensorList


class LeastSquares(Constraint):
    """[summary]

    Args:
        Constraint ([type]): [description]
    """

    def __init__(self) -> None:
        super().__init__()

    def calculate_coeffs(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """[summary]

        Args:
            sparse_thetas (TensorList): [description]
            time_derivs (TensorList): [description]

        Returns:
            [type]: [description]
        """
        opt_coeff = []
        for theta, dt in zip(sparse_thetas, time_derivs):
            Q, R = torch.qr(theta)  # solution of lst. sq. by QR decomp.
            opt_coeff.append(torch.inverse(R) @ Q.T @ dt)
        return opt_coeff
