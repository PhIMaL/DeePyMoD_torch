from itertools import product, combinations
import sys
import torch

def string_matmul(list_1, list_2):
    prod = [element[0] + element[1] for element in product(list_1, list_2)]
    return prod


def terms_definition(poly_list, deriv_list):
    theta_uv = [string_matmul(u, v) for u, v in combinations(poly_list, 2)]
    theta_dudv = [string_matmul(du, dv)[1:] for du, dv in combinations(deriv_list, 2)]
    theta_udv = [string_matmul(u[1:], dv[1:]) for u, dv in product(poly_list, deriv_list)]
    theta = [element for theta_specific in (theta_uv + theta_dudv + theta_udv) for element in theta_specific]

    return theta

def create_deriv_data(X, max_order):
    ''' Utility function to create the deriv object'''
    if max_order == 1:
        dX = (torch.eye(X.shape[1]) * torch.ones(X.shape[0])[:, None, None])[:, None, :]
    else: 
        dX = [torch.eye(X.shape[1]) * torch.ones(X.shape[0])[:, None, None]]
        dX.extend([torch.zeros_like(dX[0]) for order in range(max_order-1)])
        dX = torch.stack(dX, dim=1)
        
    return (X, dX)

