from deepymod_torch.constructors import build_network, build_coeff_vector
from deepymod_torch.training import train

import torch.nn as nn
from deepymod_torch.losses import DeepMoDLoss


class deepmod_type(nn.Module):
    def __init__(self, config, network_constructor=build_network):
        super().__init__()
        self.config = config

        self.network = network_constructor(**self.config)
        self.coeff_vector_list = build_coeff_vector(self.network, self.config['library_args']['diff_order'])

    def forward(self, input):
        prediction, time_deriv, theta = self.network(input)
        return prediction, time_deriv, theta


class DeepMod():
    def __init__(self, config):
        self.network = deepmod_type(config)

    def train(self, data, target, optimizer, max_iterations, loss_func=DeepMoDLoss, loss_func_args={'l1': 10e-5}):
        train(data, target, self.network, optimizer, max_iterations, loss_func, loss_func_args)

    def __call__(self, input):
        prediction = self.network(input)
        return prediction

    def __getattr__(self, name):
        return self.network.__dict__[name]    
    
    def parameters(self):
        return self.network.parameters()
    
    @property
    def coeff_vector_list(self):
        return self.network.coeff_vector_list
        
