import numpy as np
import torch
from torch.autograd import grad
from itertools import permutations, product


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
        du = torch.cat((du, grad(du[:, order:order+1], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 0:1]), dim=1)

    return time_deriv, du


def library_1D_in(data, prediction, library_config):
    poly_list = []
    deriv_list = []
    time_deriv_list = []

    # Creating lists for all outputs
    for output in torch.arange(prediction.shape[1]):
        time_deriv, du = library_deriv(data, prediction, library_config)
        u = library_poly(data, prediction, library_config)

        poly_list.append(u)
        deriv_list.append(du)
        time_deriv_list.append(time_deriv)

    # Calculating theta
    if len(poly_list) == 1:
        theta = torch.matmul(poly_list[0][:, :, None], deriv_list[0][:, None, :]).view(poly_list[0].shape[0], poly_list[0].shape[1] * deriv_list[0].shape[1])
    else:
        theta_uv = torch.cat([torch.matmul(u[:, :, None], v[:, None, :]).view(u.shape[0], total_terms)[:, 1:] for u, v in permutations(poly_list, 2)], 1)
        theta_dudv = torch.cat([torch.matmul(du[:, :, None], dv[:, None, :]).view(u.shape[0], total_terms)[:, 1:] for du, dv in permutations(deriv_list, 2)], 1)
        theta_udu = torch.cat([torch.matmul(u[:, :, None], du[:, None, :]).view(u.shape[0], total_terms)[:, 1:] for u, dv in product(poly_list, deriv_list)], 1)
        theta = torch.cat([torch.ones(poly_list[0].shape[0], 1, dtype=torch.float32), theta_uv, theta_dudv, theta_udu], dim=1)

    return time_deriv_list, theta


def library_1D(data, prediction,library_config):

    max_order = library_config['poly_order']
    max_diff = library_config['diff_order']

    # Calculate the polynomes of u

    y = prediction
    u = torch.ones_like(prediction)
    for order in np.arange(1,max_order+1):
        u = torch.cat((u, u[:, order-1:order]*prediction),dim=1)

    # Calculate the derivatives

    dy = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
    y_x = dy[:, 0:1]
    y_t = dy[:, 1:2]

    du = torch.ones_like(y_x)
    du = torch.cat((du, y_x.reshape(-1,1)),dim=1)

    for order in np.arange(1, max_diff):
        du = torch.cat((du, grad(du[:, order:order+1].reshape(-1,1), data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 0:1]), dim=1)

    # Constructing Theta

    theta = du
    for order in np.arange(1,max_order+1):
        theta = torch.cat((theta, u[:,order:order+1].reshape(-1,1)*du), dim=1)

    return [y_t], theta
