import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, delta=1e-2, initial_epoch=1000, sparsity_update_period=500):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.delta = delta
        self.initial_epoch = initial_epoch
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
        self.first_sparsity_epoch = 1e8
        self.sparsity_update_period = sparsity_update_period
    
    def __call__(self, epoch, val_loss, model, optimizer):
        if (epoch >= self.initial_epoch) and (self.first_sparsity_epoch == 1e8): # first part
            self.update_score(val_loss, model, optimizer, epoch)
        elif (epoch > self.first_sparsity_epoch) and ((epoch - self.first_sparsity_epoch) % self.sparsity_update_period == 0): # sparsity update
            self.early_stop = True
        else: # before initial epoch
            pass
            
    def update_score(self, val_loss, model, optimizer, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.first_sparsity_epoch = epoch
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'model_checkpoint.pt')
        torch.save(optimizer.state_dict(), 'optimizer_checkpoint.pt')
        self.val_loss_min = val_loss
        
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf