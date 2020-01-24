import torch

def scaling_single_vec(coeff_vector, sparse_theta, time_deriv):
    '''
    Rescales the weight vector according to vec_rescaled = vec * |library|/|time_deriv|.
    Columns in library correspond to elements of weight_vector.
     '''
    scaling_time = torch.norm(time_deriv, dim=0)
    scaling_theta = torch.norm(sparse_theta, dim=0)[:, None]
    coeff_vector_scaled = coeff_vector * (scaling_theta / scaling_time)

    return coeff_vector_scaled

def scaling(coeff_vector_list, sparse_theta_list, time_deriv_list):
    '''Wrapper around scaling_single_vec to scale multiple eqs. See scaling_single_vec for more details. '''
    coeff_vector_scaled_list = [scaling_single_vec(coeff_vector, sparse_theta, time_deriv) for time_deriv, sparse_theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, coeff_vector_list)]
    return coeff_vector_scaled_list

def threshold_single(coeff_vector_scaled, coeff_vector):
    '''Removes coefficient if |value| < std(coefficient_vec) and returns new coefficient vector and sparsity mask. '''
    sparse_coeff_vector = torch.where(torch.abs(coeff_vector_scaled) > torch.std(coeff_vector_scaled, dim=0), coeff_vector, torch.zeros_like(coeff_vector_scaled))
    sparsity_mask = torch.nonzero(sparse_coeff_vector)[:, 0].detach()  # detach it so it doesn't get optimized and throws an error
    sparse_coeff_vector = torch.nn.Parameter(sparse_coeff_vector[sparsity_mask].clone().detach())

    return sparse_coeff_vector, sparsity_mask
    
def threshold(coeff_vector_list, sparse_theta_list, time_deriv_list):
    '''Wrapper around threshold_single to threshold list of vectors. Also performs scaling.'''

    coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list)
    result = [threshold_single(coeff_vector_scaled, coeff_vector) for coeff_vector_scaled, coeff_vector in zip(coeff_vector_scaled_list, coeff_vector_list)]
    sparse_coeff_vector_list, sparsity_mask_list = map(list, zip(*result))
    
    return sparse_coeff_vector_list, sparsity_mask_list