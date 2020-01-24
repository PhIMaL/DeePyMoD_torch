import numpy as np
import sys, time
import torch
from torch.utils.tensorboard import SummaryWriter

class Tensorboard():
    '''Tensorboard class for logging during deepmod training. '''
    def __init__(self, number_of_terms):
        self.writer = SummaryWriter()
        self.writer.add_custom_scalars(custom_board(number_of_terms))

    def write(self, iteration, loss, loss_mse, loss_reg, loss_l1, coeff_vector_list, coeff_vector_scaled_list):
        # Logs losses, costs and coeff vectors.
        self.writer.add_scalar('Total loss', loss, iteration)
        for idx in range(len(loss_mse)):
            self.writer.add_scalar('MSE '+str(idx), loss_mse[idx], iteration)
            self.writer.add_scalar('Regression '+str(idx), loss_reg[idx], iteration)
            self.writer.add_scalar('L1 '+str(idx), loss_l1[idx], iteration)
            for element_idx, element in enumerate(torch.unbind(coeff_vector_list[idx])): # Tensorboard doesnt have vectors, so we unbind and plot them in together in custom board
                self.writer.add_scalar('coeff ' + str(idx) + ' ' + str(element_idx), element, iteration)
            for element_idx, element in enumerate(torch.unbind(coeff_vector_scaled_list[idx])):
                self.writer.add_scalar('scaled_coeff ' + str(idx) + ' ' + str(element_idx), element, iteration)

    def close(self):
        self.writer.close()

def custom_board(number_of_terms):
    '''Custom scalar board for tensorboard.'''
    number_of_eqs = len(number_of_terms)
    # Initial setup, including all the costs and losses
    custom_board = {'Costs': {'MSE': ['Multiline', ['MSE_' + str(idx) for idx in np.arange(number_of_eqs)]],
                              'Regression': ['Multiline', ['Regression_' + str(idx) for idx in np.arange(number_of_eqs)]],
                              'L1': ['Multiline', ['L1_' + str(idx) for idx in np.arange(number_of_eqs)]]},
                    'Coefficients': {},
                    'Scaled coefficients': {}}

    # Add plot of normal and scaled coefficients for each equation, containing every component in single plot.
    for idx in np.arange(number_of_eqs):
        custom_board['Coefficients']['Vector_' + str(idx)] = ['Multiline', ['coeff_' + str(idx) + '_' + str(element_idx) for element_idx in np.arange(number_of_terms[idx])]]
        custom_board['Scaled coefficients']['Vector_' + str(idx)] = ['Multiline', ['scaled_coeff_' + str(idx) + '_' + str(element_idx) for element_idx in np.arange(number_of_terms[idx])]]

    return custom_board

def progress(iteration, start_time, max_iteration, cost, MSE, PI, L1):
    '''Prints and updates progress of training cycle in command line.'''
    percent = iteration.item()/max_iteration * 100
    elapsed_time = time.time() - start_time
    time_left = elapsed_time * (max_iteration/iteration - 1) if iteration.item() != 0 else 0
    sys.stdout.write(f"\r  {iteration:>9}   {percent:>7.2f}%   {time_left:>13.0f}s   {cost:>8.2e}   {MSE:>8.2e}   {PI:>8.2e}   {L1:>8.2e} ")
    sys.stdout.flush()