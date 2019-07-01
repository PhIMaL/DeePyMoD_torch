import numpy as np
import torch.nn as nn
import torch

from library_function import *
from sparsity import scaling


def network_init(network_config):
    '''
    Builds a fully-connected NN according to specified parameters.
    '''
    # unpacking configuration
    input_dim = network_config['input_dim']
    hidden_dim = network_config['hidden_dim']
    layers = network_config['layers']
    output_dim = network_config['output_dim']

    # Building network
    network = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]  # Input layer

    for hidden_layer in np.arange(layers):  # Hidden layers
        network.append(nn.Linear(hidden_dim, hidden_dim))
        network.append(nn.Tanh())

    network.append(nn.Linear(hidden_dim, output_dim))  # Output layer
    torch_network = nn.Sequential(*network)

    return torch_network


def train(data, target, network, coeff_vector, library_config, optim_config):
    '''
    Trains deepmod NN.
    '''

    max_iterations = optim_config['max_iterations']
    l1 = optim_config['lambda']
    library_function = library_config['type']

    optimizer = torch.optim.Adam([{'params': network.parameters()}, {'params': coeff_vector}])

    # Training
    print('Epoch | Total loss | MSE | PI | L1 ')
    for iteration in np.arange(max_iterations):
        # Calculating prediction and library
        prediction = network(data)
        time_deriv, theta = library_function(data, prediction, library_config)
        f = time_deriv - theta @ coeff_vector

        # Scaling
        coeff_vector_scaled = scaling(coeff_vector, theta, time_deriv)

        # Calculating losses
        loss_MSE = torch.nn.MSELoss()(prediction, target)
        loss_PI = torch.nn.MSELoss()(f, torch.zeros_like(f))
        loss_L1 = l1 * torch.sum(torch.abs(coeff_vector_scaled))
        loss = loss_MSE + loss_PI + loss_L1

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Printing
        if iteration % 1000 == 0:
            print(iteration, "%.1E" % loss.detach().numpy(), "%.1E" % loss_MSE.detach().numpy(), "%.1E" % loss_PI.detach().numpy(), "%.1E" % loss_L1.detach().numpy())
            print(np.around(coeff_vector.detach().numpy(), decimals=2))

    return time_deriv, theta, coeff_vector


def Final_Training(data, target, optim_config, library_type, library_config, network, network_config, sparse_weight_vector, sparsity_mask):

    max_it = 5000

    # Re-initialize the optimizer
    optimizer = torch.optim.Adam([{'params':network.parameters()}, {'params': sparse_weight_vector}])

    for iteration in np.arange(max_it):

        # Calculate the predicted y-value, construct the sparse library function
        prediction = network(data)
        y_t, theta = library_poly_multi(data, prediction,library_config)

        # Use a dummy vector to flatten the coefficient vector and apply the mask (This could be done more elegantly)
        dummy = torch.zeros((library_config['total_terms']*network_config['output_dim'],1))
        dummy[sparsity_mask] = sparse_weight_vector
        dummy = dummy.reshape(library_config['total_terms'], network_config['output_dim'])
        f = y_t - theta @ dummy

        # Losses: MSE, PI and L1
        loss_MSE = torch.nn.MSELoss()(prediction, target)
        loss_PI = torch.nn.MSELoss()(f, torch.zeros_like(y_t))

        # Combine all the losses
        loss = loss_MSE + loss_PI

        # Optimizwe step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the losses during training
        if iteration == 0:
            print('Epoch | Total loss | MSE | PI ')
        if iteration % 1000 == 0:
            print(iteration, "%.1Ef" % loss.detach().numpy(), "%.1E" % loss_MSE.detach().numpy(), "%.1E" % loss_PI.detach().numpy())
            print(np.around(sparse_weight_vector.detach().numpy(),decimals=2).reshape(1,-1))

    return sparse_weight_vector, prediction


def Training_MSE(data, target, optim_config, library_type, library_config, network, network_config):

    max_it = optim_config['max_iteration']
    l1 = optim_config['lambda']

    # Initialize the weight vector and optimizer

    optimizer = torch.optim.Adam([{'params':network.parameters()}])

    for iteration in np.arange(max_it):

        prediction = network(data)
        # Combine all the losses
        loss = torch.nn.MSELoss()(prediction, target)
        y_t, theta = library_1D(data, prediction,library_config)

        # Optimizwe step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the losses during training
        if iteration == 0:
            print('Epoch | Total loss | MSE ')
        if iteration % 1000 == 0:
            print(iteration, "%.1E" % loss.detach().numpy(), "%.1E" % loss.detach().numpy())

    return prediction, network, y_t, theta


def Training_PI(data, target, optim_config, library_type, library_config, network, network_config,init_coeff):

    max_it = optim_config['max_iteration']
    l1 = optim_config['lambda']

    # Initialize the weight vector and optimizer
    weight_vector = init_coeff
    optimizer = torch.optim.Adam([{'params': weight_vector}])

    for iteration in np.arange(max_it):

        # Calculate the predicted y-value, construct the library function
        prediction = network(data)
        y_t, theta = library_type(data, prediction,library_config)
        f = y_t - theta @ weight_vector

        # Losses: MSE, PI and L1
        loss_MSE = torch.nn.MSELoss()(prediction, target)
        loss_PI = torch.nn.MSELoss()(f, torch.zeros_like(y_t))
        loss_L1 = l1*nn.L1Loss()(weight_vector,torch.zeros_like(weight_vector))

        # Combine all the losses
        loss = loss_MSE + loss_PI + loss_L1

        # Optimizwe step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the losses during training
        if iteration == 0:
            print('Epoch | Total loss | MSE | PI | L1 ')
        if iteration % 1000 == 0:
            print(iteration, "%.1E" % loss.detach().numpy(), "%.1E" % loss_MSE.detach().numpy(), "%.1E" % loss_PI.detach().numpy(), "%.1E" % loss_L1.detach().numpy())
            print(np.around(weight_vector.detach().numpy(),decimals=2))

    return y_t, theta, weight_vector
