import torch
import torch.nn as nn
from typing import List


class NN(nn.Module):
    ''' Neural network function approximator.'''
    def __init__(self, n_in: int, n_hidden: List[int], n_out: int) -> None:
        super().__init__()
        self.network = self.build_network(n_in, n_hidden, n_out)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.network(input)

    def build_network(self, n_in, n_hidden, n_out):
        network = []
        architecture = [n_in] + n_hidden + [n_out]
        for layer_i, layer_j in zip(architecture, architecture[1:]):
            network.append(nn.Linear(layer_i, layer_j))
            network.append(nn.Tanh())
        network.pop()  # get rid of last activation function
        return nn.Sequential(*network)
