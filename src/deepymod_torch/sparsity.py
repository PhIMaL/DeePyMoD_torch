import torch


def scaling(weight_vector, library, time_deriv):
    scaling_time = torch.norm(time_deriv, dim=0)
    scaling_theta = torch.norm(library, dim=0)[:, None]
    scaled_weight_vector = weight_vector * (scaling_theta / scaling_time)
    return scaled_weight_vector


def threshold(scaled_coeff_vector, coeff_vector):
    sparse_coeff_vector = torch.where(torch.abs(scaled_coeff_vector) > torch.std(scaled_coeff_vector, dim=0), coeff_vector, torch.zeros_like(scaled_coeff_vector))
    sparsity_mask = torch.nonzero(sparse_coeff_vector)[:, 0]
    sparse_coeff_vector = sparse_coeff_vector[sparsity_mask].detach()

    return scaled_coeff_vector, sparsity_mask
