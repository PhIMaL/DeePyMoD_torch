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
    time_deriv, theta, weight_vector = train(data, target, network, coeff_vector, library_config, optim_config)

    # Thresholding
    scaled_coeff_vector = scaling(weight_vector, theta, time_deriv)
    sparse_coeff_vector, sparsity_mask = threshold(scaled_coeff_vector, coeff_vector)
    print(coeff_vector, sparse_coeff_vector, sparsity_mask)

    # Final Training without L1 and with the sparsity pattern
    #sparse_weight_vector, prediction = Final_Training(data, target, optim_config, library_config, network, network_config, sparse_weight_vector, sparsity_mask)

    return sparse_coeff_vector, sparsity_mask, network

'''
def DeepMod_mse(data, target, network_config, library_type, library_config, optim_config):

    # Initiate neural network, weight_vector and optimizer:
    network = LinNetwork(network_config)

    # Training of the network
    prediction, network, y_t, theta = Training_MSE(data, target, optim_config, library_type, library_config, network, network_config)


    return prediction, network, y_t, theta


def DeepMod_pretrained_nomse(data, target, network_config, library_type, library_config, optim_config, network,init_coeff):

    # Training of the network
    y_t,theta, weight_vector = Training_PI(data, target, optim_config, library_type, library_config, network, network_config,init_coeff)


    return  y_t,theta, weight_vector


def DeepMod_single(data, target, network_config, library_type, library_config, optim_config, network,init_coeff):

    # Training of the network
    y_t,theta, weight_vector = Training(data, target, optim_config, library_type, library_config, network, network_config,init_coeff)


    return  y_t,theta, weight_vector
'''
