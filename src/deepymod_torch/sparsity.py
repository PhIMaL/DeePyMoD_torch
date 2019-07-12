import torch


def scaling(weight_vector, library, time_deriv):
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

    scaling_time = torch.norm(time_deriv, dim=0)
    scaling_theta = torch.norm(library, dim=0)[:, None]
    scaled_weight_vector = weight_vector * (scaling_theta / scaling_time)
    return scaled_weight_vector


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
