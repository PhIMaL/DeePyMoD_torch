import torch

def scaling_single_vec(coeff_vector, sparse_theta, time_deriv):
    scaling_time = torch.norm(time_deriv, dim=0)
    scaling_theta = torch.norm(sparse_theta, dim=0)[:, None]
    coeff_vector_scaled = coeff_vector * (scaling_theta / scaling_time)

    return coeff_vector_scaled

def scaling(coeff_vector_list, sparse_theta_list, time_deriv_list):
    '''
    Rescales the weight vector according to vec_rescaled = vec * |library|/|time_deriv|.
    Columns in library correspond to elements of weight_vector.

    Parameters
    ----------
    weight_vector : tensor of size (Mx1).
        The weight vector to be rescaled.
    library : tensor of size (NxM)
        The library matrix used to rescale weight_vector.
    time_deriv : tensor of size (Nx1)
        The time derivative vector used to rescale weight_vector.

    Returns
    -------
    tensor of size (Mx1)
        Rescaled weight vector.
    '''
    coeff_vector_scaled_list = [scaling_single_vec(coeff_vector, sparse_theta, time_deriv) for time_deriv, sparse_theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, coeff_vector_list)]
    return coeff_vector_scaled_list

def threshold_single(coeff_vector_scaled, coeff_vector):
    sparse_coeff_vector = torch.where(torch.abs(coeff_vector_scaled) > torch.std(coeff_vector_scaled, dim=0), coeff_vector, torch.zeros_like(coeff_vector_scaled))
    sparsity_mask = torch.nonzero(sparse_coeff_vector)[:, 0].detach()  # detach it so it doesn't get optimized and throws an error
    sparse_coeff_vector = torch.nn.Parameter(sparse_coeff_vector[sparsity_mask].clone().detach())  # so it can be optimized

    return sparse_coeff_vector, sparsity_mask
    
def threshold(coeff_vector_list, sparse_theta_list, time_deriv_list):
    coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list)
    result = [threshold_single(coeff_vector_scaled, coeff_vector) for coeff_vector_scaled, coeff_vector in zip(coeff_vector_scaled_list, coeff_vector_list)]
    sparse_coeff_vector_list, sparsity_mask_list = map(list, zip(*result))
    
    return sparse_coeff_vector_list, sparsity_mask_list