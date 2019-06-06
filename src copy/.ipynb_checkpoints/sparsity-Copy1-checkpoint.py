import numpy as np
import torch.nn as nn
import torch

def scaling(y_t, theta, weight_vector):
    scaling_time = torch.norm(y_t)
    scaling_theta = torch.norm(theta, dim=0)[:, None]
    scaled_weight_vector = weight_vector * (scaling_theta / scaling_time)    
    return scaled_weight_vector

def threshold(scaled_weight_vector, weight_vector):
    sparsity  = torch.where(torch.abs(scaled_weight_vector) > torch.tensor(0.01), scaled_weight_vector, torch.zeros_like(scaled_weight_vector))
    print(sparsity)
    sparsity_mask  = torch.nonzero(sparsity)
    print(sparsity_mask)
    
    sparse_weight_vector = weight_vector[sparsity_mask[:,0]].reshape(-1,sparsity.shape[0]).clone().detach().requires_grad_(True)
    print(sparse_weight_vector)
    sparsity_mask.reshape(2,2)
    return sparse_weight_vector, sparsity_mask[:,0]

# torch.std(scaled_weight_vector)