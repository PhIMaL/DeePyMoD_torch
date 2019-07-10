import numpy as np


def custom_board(coeff_vector_list):
    custom_board = {'Costs': {'MSE': ['Multiline', ['MSE_' + str(idx) for idx in np.arange(len(coeff_vector_list))]],
                              'Regression': ['Multiline', ['Regression_' + str(idx) for idx in np.arange(len(coeff_vector_list))]],
                              'L1': ['Multiline', ['L1_' + str(idx) for idx in np.arange(len(coeff_vector_list))]]},
                    'Coefficients': {}, 
                    'Scaled coefficients': {}}

    for idx in np.arange(len(coeff_vector_list)):
        custom_board['Coefficients']['Vector_' + str(idx)] = ['Multiline', ['coeff_' + str(idx) + '_' + str(element_idx) for element_idx in np.arange(coeff_vector_list[idx].shape[0])]]
        custom_board['Scaled coefficients']['Vector_' + str(idx)] = ['Multiline', ['scaled_coeff_' + str(idx) + '_' + str(element_idx) for element_idx in np.arange(coeff_vector_list[idx].shape[0])]]

    return custom_board
    