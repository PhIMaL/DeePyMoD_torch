import numpy as np
import torch.nn as nn
import torch


def scaling(weight_vector, library, time_deriv):
    scaling_time = torch.norm(time_deriv, dim=0)
    scaling_theta = torch.norm(library, dim=0)[:, None]
    scaled_weight_vector = weight_vector * (scaling_theta / scaling_time)
    return scaled_weight_vector


def threshold(scaled_weight_vector, weight_vector):
    sparsity  = torch.where(torch.abs(scaled_weight_vector) > torch.std(scaled_weight_vector, dim=0), scaled_weight_vector, torch.zeros_like(scaled_weight_vector))
    print('std',torch.std(scaled_weight_vector, dim=0))
    print('sparsity',sparsity)
    sparsity_mask  = torch.nonzero(torch.reshape(sparsity,(1,-1)))[:,1]

    print('sparsity mask',sparsity_mask)
    sparse_weight_vector = weight_vector.reshape(-1,1).clone().detach().requires_grad_(True)

    print('sparsity weight vector',sparse_weight_vector)
    sparse_weight_vector = sparse_weight_vector[sparsity_mask]
    print('sparsity weight vector',sparse_weight_vector)
    sparse_weight_vector = sparse_weight_vector.reshape(-1,1).clone().detach().requires_grad_(True)


    return sparse_weight_vector, sparsity_mask

# torch.std(scaled_weight_vector)
