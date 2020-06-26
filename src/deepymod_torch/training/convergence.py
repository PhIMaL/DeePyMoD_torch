import torch
'''This module implements convergence criteria'''

class Convergence:
    '''Implements convergence criterium. Convergence is when change in patience
    epochs is smaller than delta.
    '''
    def __init__(self, patience=100, delta=0.05):
        '''
        Initializes
        '''
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.start_l1 = None
        self.converged = False

    def __call__(self, epoch, l1_norm):
        '''
        Calling
        '''
        if self.start_l1 is None:
            self.start_l1 = l1_norm
        elif torch.abs(self.start_l1 - l1_norm).item() < self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.converged = True
        else:
            self.start_l1 = l1_norm
            self.counter = 0
        