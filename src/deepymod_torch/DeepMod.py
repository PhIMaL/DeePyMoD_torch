import torch
import torch.nn as nn
from deepymod_torch.training import train, train_mse, train_deepmod
from deepymod_torch.network import Fitting, Library, Linear, Tanh


class DeepMod(nn.Module):
    ''' Class based interface for deepmod.'''
    def __init__(self, config):
        super().__init__()
        self.network = build_network(**config)

    def train(self, data, target, optimizer, max_iterations, type='single_cycle', loss_func_args={'l1':10**-5}):
        if type == 'mse':
            train_mse(data, target, self.network, optimizer, max_iterations, loss_func_args={}) # Trains only mse.
        elif type == 'single_cycle':
            train(data, target, self.network, optimizer, max_iterations, loss_func_args) #DeepMod style training, but doesn't threshold.
        elif type == 'deepmod':
            train_deepmod(data, target, self.network, optimizer, max_iterations, loss_func_args) # Does full deepmod cycle.

    def forward(self, input):
        output = self.network(input)
        return output

    # Properties below implemented for easy access
    @property
    def coeff_vector_list(self):
        return self.network[-1].coeff_vector_list
    
    @property
    def sparsity_mask_list(self):
        return self.network[-1].sparsity_mask_list


def build_network(input_dim, hidden_dims, output_dim, library_function, library_args):
    ''' Build deepmod model.'''
    network = []
    hs = [input_dim] + hidden_dims + [output_dim]
    for h0, h1 in zip(hs, hs[1:]):  # Hidden layers
        network.append(Linear(h0, h1))
        network.append(Tanh())
    network.pop()
    print(network)

    network.append(Library(library_function, library_args)) # Library layer

    # Do a fake forward pass to infer number of terms from library
    input = torch.ones((1, input_dim), dtype=torch.float32, requires_grad=True)
    for layer in network:
        input = layer(input)
    n_terms = input[2].shape[1]
    network.append(Fitting(n_terms, output_dim))
    torch_network = nn.Sequential(*network)

    return torch_network



