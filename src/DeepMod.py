import numpy as np
import torch.nn as nn
import torch

from library_function import *
from neural_net import *
from sparsity import scaling, threshold


def DeepMod(data, target, network_config, library_type, library_config, optim_config, init_coef):   
    
    # Initiate neural network, weight_vector and optimizer:
    network = LinNetwork(network_config)    
    
    # Training of the network
    y_t,theta, weight_vector = Training(data, target, optim_config, library_type, library_config, network, network_config, init_coef)
    
    # Scaling
    scaled_weight_vector = scaling(y_t,theta, weight_vector)
    print(scaled_weight_vector)

    # Thresholding
    sparse_weight_vector, sparsity_mask =  threshold(scaled_weight_vector,weight_vector)    
    print(sparsity_mask)
    
    # Final Training without L1 and with the sparsity pattern
    sparse_weight_vector, prediction = Final_Training(data, target, optim_config, library_config, network, network_config, sparse_weight_vector, sparsity_mask)
    
    return sparse_weight_vector, sparsity_mask, prediction, network


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