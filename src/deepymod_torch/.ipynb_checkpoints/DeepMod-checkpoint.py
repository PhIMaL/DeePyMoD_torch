from deepymod_torch.neural_net import deepmod_init, train
from deepymod_torch.sparsity import scaling, threshold


def DeepMoD(data, target, network_config, library_config, optim_config):
    '''
    Runs the deepmod algorithm on the supplied dataset. Mostly a convenience function and can be used as
    a basis for more complex training means. First column of data must correspond to time coordinates, spatial coordinates
    are second column and up. Diagnostics are written to runs/ directory, can be analyzed with tensorboard.

    Parameters
    ----------
    data : Tensor of size (N x M)
        Coordinates corresponding to target data. First column must be time.
    target : Tensor of size (N x L)
        Data the NN is supposed to learn.
    network_config : dict
        Dict containing parameters for the network: {'input_dim': , 'hidden_dim': , 'layers': ,'output_dim':}
            input_dim : number of input neurons, should be same as second dimension of data.
            hiddem_dim : number of neurons in each hidden layer.
            layers : number of hidden layers.
            output_dim : number of output neurons, should be same as second dimension of data.
    library_config : dict
        Dict containing parameters for the library function: {'type': , **kwargs}
            type : library function to be used.
            kwargs : arguments to be used in the selected library function.
    optim_config : dict
        Dict containing parameters for training: {'lambda': , 'max_iterations':}
            lambda : value of l1 constant.
            max_iterations : maximum number of iterations used for training.
    Returns
    -------
    coeff_vector_list
        List of tensors containing remaining values of weight vector.
    sparsity_mask_list
        List of tensors corresponding to the maintained components of the coefficient vectors. Each list entry is one equation.
    network : pytorch sequential model
        The trained neural network.
    '''

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
