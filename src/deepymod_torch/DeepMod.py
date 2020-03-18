import torch
import torch.nn as nn
from deepymod_torch.network import Fitting, Library


class DeepMod(nn.Module):
    ''' Class based interface for deepmod.'''
    def __init__(self, n_in, hidden_dims, n_out, library_function, library_args):
        super().__init__()
        self.network = self.build_network(n_in, hidden_dims, n_out)
        self.library = Library(library_function, library_args)
        self.fit = self.build_fit_layer(n_in, n_out, library_function, library_args)

    def forward(self, input):
        prediction = self.network(input)
        time_deriv, theta = self.library((prediction, input))
        sparse_theta, coeff_vector = self.fit(theta)
        return prediction, time_deriv, sparse_theta, coeff_vector

    def build_network(self, n_in, hidden_dims, n_out):
        # NN
        network = []
        hs = [n_in] + hidden_dims + [n_out]
        for h0, h1 in zip(hs, hs[1:]):  # Hidden layers
            network.append(nn.Linear(h0, h1))
            network.append(nn.Tanh())
        network.pop()  # get rid of last activation function
        network = nn.Sequential(*network) 

        return network

    def build_fit_layer(self, n_in, n_out, library_function, library_args):
        sample_input = torch.ones((1, n_in), dtype=torch.float32, requires_grad=True)
        n_terms = self.library((self.network(sample_input), sample_input))[1].shape[1] # do sample pass to infer shapes
        fit_layer = Fitting(n_terms, n_out)

        return fit_layer

    # Function below make life easier
    def network_parameters(self):
        return self.network.parameters()

    def coeff_vector(self):
        return self.fit.coeff_vector.parameters()
