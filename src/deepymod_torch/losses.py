import torch
import torch.nn as nn
from deepymod_torch.sparsity import scaling


class RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, coeff_vector_list):
        prediction, time_deriv_list, sparse_theta_list = input
        loss = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        return loss


def reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list):
    loss = torch.stack([torch.mean((time_deriv - theta @ coeff_vector)**2) for time_deriv, theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, coeff_vector_list)])
    return loss


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        loss = mse_loss(prediction, target)
        return loss


def mse_loss(prediction, target):
    loss = torch.mean((prediction - target)**2, dim=0)
    return loss


class L1Loss(nn.Module):
    def __init__(self, l1):
        super().__init__()
        self.l1 = l1

    def forward(self, coeff):
        loss = l1_loss(coeff, self.l1)
        return loss


def l1_loss(coeff_vector_list, l1):
    loss = torch.stack([torch.sum(torch.abs(coeff_vector)) for coeff_vector in coeff_vector_list])
    return l1 * loss


class DeepMoDLoss(nn.Module):
    def __init__(self, l1):
        super().__init__()
        self.l1 = l1

    def forward(self, input, target, coeff_vector_list):
        prediction, time_deriv_list, sparse_theta_list = input

        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_mse = mse_loss(prediction, target)

        coeff_vector_scaled_list = [scaling(coeff_vector, sparse_theta, time_deriv) for time_deriv, sparse_theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, coeff_vector_list)]
        loss_l1 = l1_loss(coeff_vector_scaled_list, self.l1)

        loss = torch.sum(loss_reg) + torch.sum(loss_mse) + torch.sum(loss_l1)
        return loss



 
  
