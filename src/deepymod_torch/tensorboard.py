import numpy as np


def custom_board(coeff_vector_list):
    '''
    Constructs a dict containing the layout for a custom scalar board for tensorboard. Shapes and amount of plots are inferred from the
    coefficient vector list.

    Parameters
    ----------
    coeff_vector_list : tensor list
        list of coefficient vectors.

    Returns
    -------
    custom_board : dict
        dict containing layout of custom board.
    '''

    # Initial setup, including all the costs and losses
    custom_board = {'Costs': {'MSE': ['Multiline', ['MSE_' + str(idx) for idx in np.arange(len(coeff_vector_list))]],
                              'Regression': ['Multiline', ['Regression_' + str(idx) for idx in np.arange(len(coeff_vector_list))]],
                              'L1': ['Multiline', ['L1_' + str(idx) for idx in np.arange(len(coeff_vector_list))]]},
                    'Coefficients': {},
                    'Scaled coefficients': {}}

    # Add plot of normal and scaled coefficients for each equation, containing every component in single plot.
    for idx in np.arange(len(coeff_vector_list)):
        custom_board['Coefficients']['Vector_' + str(idx)] = ['Multiline', ['coeff_' + str(idx) + '_' + str(element_idx) for element_idx in np.arange(coeff_vector_list[idx].shape[0])]]
        custom_board['Scaled coefficients']['Vector_' + str(idx)] = ['Multiline', ['scaled_coeff_' + str(idx) + '_' + str(element_idx) for element_idx in np.arange(coeff_vector_list[idx].shape[0])]]

    return custom_board
