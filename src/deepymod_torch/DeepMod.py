import torch
import torch.nn as nn
from deepymod_torch.training import train, train_mse, train_deepmod
from deepymod_torch.network import Linear, Tanh
from deepymod_torch.utilities import create_deriv_data

class DeepMod(nn.Module):
    '''
    Class based interface to Deepmod model. Just encapsulates different training regimes.

    Parameters
    ----------
    config : dict
        dict containing configuration for DeepMod model. See example for dict keywords.

    Returns
    -------
    DeepMoD : class instance
        class instance of DeepMod type.
    '''
    def __init__(self, config):
        super().__init__()
        self.network = build_network(**config)

    def train(self, data, target, optimizer, max_iterations, type='single_cycle', loss_func_args={}):
        if type == 'mse':
            train_mse(data, target, self.network, optimizer, max_iterations, loss_func_args={}) # Trains only mse.
        elif type == 'single_cycle':
            train(data, target, self.network, optimizer, max_iterations, loss_func_args={'l1':1e-5}) #DeepMod style training, but doesn't threshold.
        elif type == 'deepmod':
            train_deepmod(data, target, self.network, optimizer, max_iterations, loss_func_args={'l1':1e-5}) # Does full deepmod cycle.

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


def build_network(input_dim, hidden_dim, layers, output_dim, library_function, library_args):
    '''
    Constructs the DeepMoD model. 

    Parameters
    ----------
    input_dim : int
        Number of neurons in input layer of NN.
    hidden_dim : int
        Number of neurons in hidden layers of NN.
    layers : int
        Number of hidden layers in NN.
    output_dim : int
        Number of neurons in output of NN.

    Returns
    -------
    torch_network : class instance
        class instance of pytorch network.
    '''
    network = [Linear(input_dim, hidden_dim), Tanh()]  # Input layer
    for hidden_layer in torch.arange(layers):  # Hidden layers
        network.append(Linear(hidden_dim, hidden_dim))
        network.append(Tanh())
    network.append(Linear(hidden_dim, output_dim))  # Output layer
    
    network.append(library_function(input_dim, output_dim, **library_args)) # Library layer
    torch_network = nn.Sequential(*network)

    return torch_network



