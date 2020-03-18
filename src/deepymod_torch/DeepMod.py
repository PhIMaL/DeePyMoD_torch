import torch
import torch.nn as nn
from deepymod_torch.training import train, train_mse, train_deepmod
from deepymod_torch.network import Fitting, Library


class DeepMod(nn.Module):
    ''' Class based interface for deepmod.'''
    def __init__(self, config):
        super().__init__()
        self.network, self.library, self.fit = self.build_network(**config)

    def forward(self, input):
        prediction = self.network(input)
        time_deriv, theta = self.library((prediction, input))
        sparse_theta, coeff_vector = self.fit(theta)
        return prediction, time_deriv, sparse_theta, coeff_vector

    def build_network(self, input_dim, hidden_dims, output_dim, library_function, library_args):
        # NN
        network = []
        hs = [input_dim] + hidden_dims + [output_dim]
        for h0, h1 in zip(hs, hs[1:]):  # Hidden layers
            network.append(nn.Linear(h0, h1))
            network.append(nn.Tanh())
        network.pop()  # get rid of last activation function
        network = nn.Sequential(*network) 

        # Library layer
        library = Library(library_function, library_args)

        # Fitting layer
        sample_input = torch.ones((1, input_dim), dtype=torch.float32, requires_grad=True)
        sample_output = network(sample_input)
        n_terms = library((sample_output, sample_input))[1].shape[1]
        fit = Fitting(n_terms, output_dim)

        return network, library, fit
