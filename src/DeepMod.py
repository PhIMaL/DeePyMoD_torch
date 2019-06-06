import numpy as np
import torch.nn as nn
import torch

from library_function import library_1D
from neural_net import LinNetwork, Training, Final_Training
from sparsity import scaling, threshold

def DeepMod(data, target, network_config, library_config, optim_config):   
    
    # Initiate neural network, weight_vector and optimizer:
    network = LinNetwork(network_config)    
    
    # Training of the network
    y_t,theta, weight_vector = Training(data, target, optim_config, library_config, network, network_config)
    
    # Scaling
    scaled_weight_vector = scaling(y_t,theta, weight_vector)
    print(scaled_weight_vector)
    # Thresholding
    sparse_weight_vector, sparsity_mask =  threshold(scaled_weight_vector,weight_vector)    
    print(sparsity_mask)
    
    # Final Training without L1 and with the sparsity pattern
    sparse_weight_vector, prediction = Final_Training(data, target, optim_config, library_config, network, sparse_weight_vector, sparsity_mask)
    
    return sparse_weight_vector, sparsity_mask, prediction, network

