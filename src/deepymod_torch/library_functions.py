import torch
from itertools import combinations, product


def basic_library_function(input, poly_order, diff_order):
    '''Calculates the library and time deriv from NN output. See https://github.com/PhIMaL/network_derivs for additional info on how derivs are calculated.'''
    X, dX = input # The network now returns the (output, deriv) tuple. dX is 4-dimensional tensor of derivatives with each axis the following meaning: (sample, order, input, output)
    samples = X.shape[0]

    # Time derivatives
    dt = dX[:, 0, :1, :] # time is first input and we only need first order so dX[:, 0, :1, :]
    time_deriv_list = torch.unbind(dt, dim=2)

    # Polynomial part
    u = torch.ones_like(X)[:, None, :]
    for order in torch.arange(1, poly_order+1):
        u = torch.cat((u, u[:, order-1:order, :] * X[:, None, :]), dim=1)
    poly_list = torch.unbind(u, dim=2) #list with each entry corresponding to eq.

    # Derivative part
    dx = dX[:, :, 1:, :]  # spatial are all inputs after 1 so dX[:, :, 1:, :]
    deriv_list = [torch.cat((torch.ones((samples, 1)), eq.reshape(samples, -1)), dim=1) for eq in torch.unbind(dx, dim=3)] #list with each entry corresponding to eq.
        
    # Combining to make  theta
    if len(poly_list) == 1:
        theta = torch.matmul(poly_list[0][:, :, None], deriv_list[0][:, None, :]).reshape(samples, -1) # If we have a single output, we simply calculate and flatten matrix product between polynomials and derivatives to get library
    else:
        theta_uv = torch.cat([torch.matmul(u[:, :, None], v[:, None, :]).reshape(samples, -1) for u, v in combinations(poly_list, 2)], 1)  # calculate all unique combinations between polynomials
        theta_dudv = torch.cat([torch.matmul(du[:, :, None], dv[:, None, :]).reshape(samples, -1)[:, 1:] for du, dv in combinations(deriv_list, 2)], 1) # calculate all unique combinations of derivatives
        theta_udu = torch.cat([torch.matmul(u[:, 1:, None], du[:, None, 1:]).reshape(samples, -1) for u, du in product(poly_list, deriv_list)], 1)  # calculate all unique products of polynomials and derivatives            theta = torch.cat([theta_uv, theta_dudv, theta_udu], dim=1)

    return time_deriv_list, theta


