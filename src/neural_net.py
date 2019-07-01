import numpy as np
import torch.nn as nn
import torch

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


def train(data, target, network, coeff_vector, sparsity_mask, library_config, optim_config):
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
        sparse_theta = theta[:, sparsity_mask]
        f = time_deriv - sparse_theta @ coeff_vector

        # Scaling
        coeff_vector_scaled = scaling(coeff_vector, sparse_theta, time_deriv)

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
