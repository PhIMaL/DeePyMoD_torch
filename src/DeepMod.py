import numpy as np
import torch.nn as nn
import torch

from neural_net import network_init, train
from sparsity import scaling, threshold


def DeepMoD(data, target, network_config, library_config, optim_config):
    # Initiating
    network = network_init(network_config)
    coeff_vector = torch.randn((library_config['total_terms'], 1), dtype=torch.float32, requires_grad=True)

    # Training of the network
    sparsity_mask = torch.arange(library_config['total_terms'])
    time_deriv, theta, weight_vector = train(data, target, network, coeff_vector, sparsity_mask, library_config, optim_config)

    # Thresholding
    scaled_coeff_vector = scaling(weight_vector, theta, time_deriv)
    sparse_coeff_vector, sparsity_mask = threshold(scaled_coeff_vector, coeff_vector)
    print(coeff_vector, sparse_coeff_vector, sparsity_mask)

    # Final Training without L1 and with the sparsity pattern
    time_deriv, theta, weight_vector = train(data, target, network, sparse_coeff_vector, sparsity_mask, library_config, optim_config)

    return sparse_coeff_vector, sparsity_mask, network
