import torch
import torch.nn as nn

from deepymod_torch.nn import Linear, Tanh, create_deriv_data


def build_network(input_dim, hidden_dim, layers, output_dim):
    network = [Linear(input_dim, hidden_dim), Tanh()]  # Input layer

    for hidden_layer in torch.arange(layers):  # Hidden layers
        network.append(Linear(hidden_dim, hidden_dim))
        network.append(Tanh())

    network.append(Linear(hidden_dim, output_dim))  # Output layer
    torch_network = nn.Sequential(*network)

    return torch_network


def build_coeff_vector(network, library_function, max_order):
    input_dim = next(network.parameters()).shape[1]
    sample_data = create_deriv_data(torch.ones(1, input_dim), max_order)# we run a single forward pass on fake data to infer shapes
    sample_prediction = network(sample_data)
    theta = library_function(sample_prediction)[1]
    total_terms = theta.shape[1]

    coeff_vector_list = nn.ParameterList([torch.nn.Parameter(torch.rand((total_terms, 1), dtype=torch.float32)) for _ in torch.arange(sample_prediction[0].shape[1])])
        
    return coeff_vector_list
