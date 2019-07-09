import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations, product


def library_poly(data, prediction, library_config):
    max_order = library_config['poly_order']

    # Calculate the polynomes of u
    u = torch.ones_like(prediction)
    for order in np.arange(1, max_order+1):
        u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

    return u


def library_deriv(data, prediction, library_config):
    max_order = library_config['diff_order']

    dy = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    time_deriv = dy[:, 0:1]
    du = torch.cat((torch.ones_like(time_deriv), dy[:, 1:2]), dim=1)

    for order in np.arange(1, max_order):
        du = torch.cat((du, grad(du[:, order:order+1], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 1:2]), dim=1)

    return time_deriv, du


def library_1D_in(data, prediction, library_config):
    poly_list = []
    deriv_list = []
    time_deriv_list = []

    # Creating lists for all outputs
    for output in torch.arange(prediction.shape[1]):
        time_deriv, du = library_deriv(data, prediction[:, output:output+1], library_config)
        u = library_poly(data, prediction[:, output:output+1], library_config)

        poly_list.append(u)
        deriv_list.append(du)
        time_deriv_list.append(time_deriv)

    samples = time_deriv_list[0].shape[0]
    total_terms = poly_list[0].shape[1] * deriv_list[0].shape[1]

    # Calculating theta
    if len(poly_list) == 1:
        theta = torch.matmul(poly_list[0][:, :, None], deriv_list[0][:, None, :]).view(samples, total_terms)
    else:
        theta_uv = torch.cat([torch.matmul(u[:, :, None], v[:, None, :]).view(samples, total_terms) for u, v in combinations(poly_list, 2)], 1)
        theta_dudv = torch.cat([torch.matmul(du[:, :, None], dv[:, None, :]).view(samples, total_terms)[:, 1:] for du, dv in combinations(deriv_list, 2)], 1)
        theta_udu = torch.cat([torch.matmul(u[:, 1:, None], du[:, None, 1:]).view(samples, (poly_list[0].shape[1]-1) * (deriv_list[0].shape[1]-1)) for u, dv in product(poly_list, deriv_list)], 1)
        theta = torch.cat([theta_uv, theta_dudv, theta_udu], dim=1)
    return time_deriv_list, theta


def library_1D(data, prediction, library_config):
    max_order = library_config['poly_order']
    max_diff = library_config['diff_order']

    u = torch.ones_like(prediction)
    for order in np.arange(1, max_order+1):
        u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

    # Calculate the derivatives
    dy = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    y_t = dy[:, 0:1]
    y_x = dy[:, 1:2]

    du = torch.cat((torch.ones_like(y_x), y_x), dim=1)
    for order in np.arange(2, max_diff+1):
        du = torch.cat((du, grad(du[:, order-1:order], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 1:2]), dim=1)

    # Constructing Theta
    theta = torch.matmul(u[:, :, None], du[:, None, :])
    theta = theta.view(theta.shape[0], theta.shape[1] * theta.shape[2])

    return [y_t], theta
