from deepymod_torch.neural_net import deepmod_init, train
from deepymod_torch.sparsity import scaling, threshold


def DeepMoD(data, target, network_config, library_config, optim_config):
    optim_config_internal = optim_config.copy()
    # Initiating
    network, coeff_vector_list, sparsity_mask_list = deepmod_init(network_config, library_config)
    
    # Training of the network
    time_deriv_list, theta, coeff_vector_list = train(data, target, network, coeff_vector_list, sparsity_mask_list, library_config, optim_config_internal)
    
    # Thresholding
    scaled_coeff_vector_list = [scaling(coeff_vector, theta, time_deriv) for coeff_vector, time_deriv in zip(coeff_vector_list, time_deriv_list)]
    sparse_coeff_vector_list, sparsity_mask_list = zip(*[threshold(scaled_coeff_vector, coeff_vector) for scaled_coeff_vector, coeff_vector in zip(scaled_coeff_vector_list, coeff_vector_list)])
    
    print(coeff_vector_list, sparse_coeff_vector_list, sparsity_mask_list)
    print('Now running final cycle.')

    # Final Training without L1 and with the sparsity pattern
    optim_config_internal['lambda'] = 0
    time_deriv_list, theta, coeff_vector_list = train(data, target, network, sparse_coeff_vector_list, sparsity_mask_list, library_config, optim_config_internal)

    return coeff_vector_list, sparsity_mask_list, network
    
