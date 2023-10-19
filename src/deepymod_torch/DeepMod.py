import torch
import torch.nn as nn
from deepymod_torch.network import Fitting, Library


class DeepMod(nn.Module):
    '''A PyTorch module for data-driven discovery of partial differential equations.

    This module implements a neural network architecture for discovering the governing equations of a system
    from data. The architecture consists of a fully connected neural network followed by a library of candidate
    functions and a sparse regression layer. The library of candidate functions is defined by a user-provided
    function and its arguments.

    Args:
        n_in (int): Number of input features.
        hidden_dims (list of int): List of dimensions for the hidden layers of the neural network.
        n_out (int): Number of output features.
        library_function (callable): Function that generates the library of candidate functions.
        library_args (tuple or dict): Arguments to pass to the library function.

    Attributes:
        network (nn.Sequential): The fully connected neural network.
        library (Library): The library of candidate functions.
        fit (Fitting): The sparse regression layer.
    '''
    def __init__(self, n_in, hidden_dims, n_out, library_function, library_args):
        super().__init__()
        self.network = self.build_network(n_in, hidden_dims, n_out)
        self.library = Library(library_function, library_args)
        self.fit = self.build_fit_layer(n_in, n_out, library_function, library_args)

    def forward(self, input):
        """
        Computes the forward pass of the DeepMoD model.

        Args:
            input (torch.Tensor): Input tensor (typically X_train) of shape (batch_size, input_dim).

        Returns:
            tuple: A tuple containing:
                - prediction (torch.Tensor): Output tensor of shape (batch_size, output_dim).
                - time_deriv (torch.Tensor): Time derivative tensor of shape (batch_size, output_dim).
                - sparse_theta (torch.Tensor): Sparse theta tensor of shape (n_terms, input_dim).
                - coeff_vector (torch.Tensor): Coefficient vector tensor of shape (n_terms, output_dim).
        """
        prediction = self.network(input)
        time_deriv, theta = self.library((prediction, input))
        sparse_theta, coeff_vector = self.fit(theta)
        return prediction, time_deriv, sparse_theta, coeff_vector

    def build_network(self, n_in, hidden_dims, n_out):
        """
        Builds a neural network with the specified number of input, hidden, and output nodes.

        Args:
            n_in (int): Number of input nodes.
            hidden_dims (list): List of integers specifying the number of nodes in each hidden layer.
            n_out (int): Number of output nodes.

        Returns:
            network (nn.Sequential): A PyTorch sequential neural network object.
        """  
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
        """
        Builds and returns a Fitting layer for the DeepMoD model.

        Args:
            n_in (int): Number of input features.
            n_out (int): Number of output features.
            library_function (callable): Function that generates the library.
            library_args (dict): Arguments to pass to the library function.

        Returns:
            Fitting: A Fitting layer with the appropriate number of terms for the given input and output sizes.
        """
        sample_input = torch.ones((1, n_in), dtype=torch.float32, requires_grad=True)
        n_terms = self.library((self.network(sample_input), sample_input))[1].shape[1] # do sample pass to infer shapes
        fit_layer = Fitting(n_terms, n_out)

        return fit_layer

    # Function below make life easier
    def network_parameters(self):
        return self.network.parameters()

    def coeff_vector(self):
        return self.fit.coeff_vector.parameters()
