import torch
import numpy as np


class Periodic:
    '''Controls when to apply sparsity. Initial_epoch is first time of appliance,
    then every periodicity epochs.
    '''
    def __init__(self, initial_epoch=1000, periodicity=100):
        self.initial_epoch = initial_epoch
        self.periodicity = periodicity

        self.apply_sparsity = False

    def __call__(self, iteration, l1_norm):
        if iteration >= self.initial_epoch:
            if (iteration - self.initial_epoch) % self.periodicity == 0:
                self.apply_sparsity = True
    
    def reset(self):
        self.apply_sparsity = False
