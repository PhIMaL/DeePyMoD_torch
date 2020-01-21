import torch


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
    def scaling_single_vec(coeff_vector, sparse_theta, time_deriv):
        scaling_time = torch.norm(time_deriv, dim=0)
        scaling_theta = torch.norm(sparse_theta, dim=0)[:, None]
        scaled_coeff_vector = coeff_vector * (scaling_theta / scaling_time)

        return scaled_coeff_vector

    coeff_vector_scaled = [scaling_single_vec(coeff_vector, sparse_theta, time_deriv) for time_deriv, sparse_theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, coeff_vector_list)]
    return coeff_vector_scaled

def threshold(scaled_coeff_vector, coeff_vector):
    '''
    Performs thresholding of coefficient vector based on the scaled coefficient vector.
    Components greater than the standard deviation of scaled coefficient vector are maintained, rest is set to zero.
    Also returns the location of the maintained components.

    Parameters
    ----------
    scaled_coeff_vector : tensor of size (Mx1)
        scaled coefficient vector, used to determine thresholding.
    coeff_vector : tensor of size (Mx1)
        coefficient vector to be thresholded.

    Returns
    -------
    tensor of size (Nx1)
        vector containing remaining values of weight vector.
    tensor of size(N)
        tensor containing index location of non-zero components.
    '''
    sparse_coeff_vector = torch.where(torch.abs(scaled_coeff_vector) > torch.std(scaled_coeff_vector, dim=0), coeff_vector, torch.zeros_like(scaled_coeff_vector))
    sparsity_mask = torch.nonzero(sparse_coeff_vector)[:, 0].detach()  # detach it so it doesn't get optimized and throws an error
    sparse_coeff_vector = sparse_coeff_vector[sparsity_mask].clone().detach().requires_grad_(True)  # so it can be optimized

    return sparse_coeff_vector, sparsity_mask
