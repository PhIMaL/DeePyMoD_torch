import torch
from deepymod_torch.network import Library
from itertools import combinations, product


class library_1D(Library):
    def __init__(self, input_dim, output_dim, diff_order, poly_order):
        self.poly_order = poly_order
        super().__init__(input_dim, output_dim, diff_order)

    def theta(self, input):
        X, dX = input
        dt = dX[:, 0, 0:1, 0]
        dx = dX[:, :, 1, 0]

        # Calculate the polynomes of u
        u = torch.ones_like(X)
        for order in torch.arange(1, self.poly_order+1):
            u = torch.cat((u, u[:, order-1:order] * X), dim=1)

        # Calculate derivs
        du = torch.cat((torch.ones((dx.shape[0], 1)), dx), dim=1)

        theta = (u[:, :, None] @ du[:, None, :]).reshape(u.shape[0], -1)

        return [dt], theta


def library_1D_in(data, prediction, library_config):
    '''
    Calculates a library function for a 1D+1 input for M coupled equations consisting of all polynomials up to order K and derivatives up to order
    L and all possible combinations (i.e. combining two terms) of these.

    Parameters
    ----------
    data : tensor of size (N x 2)
        coordinates to whose respect the derivatives of prediction are calculated. First column is time, space second column.
    prediction : tensor of size (N x M)
        dataset from which the library is constructed.
    library_config : dict
        dictionary containing options for the library function.

    Returns
    -------
    time_deriv_list : tensor list of length of M
        list containing the time derivatives, each entry corresponds to an equation.
    theta : tensor
        library matrix tensor.
    '''
    poly_list = []
    deriv_list = []
    time_deriv_list = []

    # Creating lists for all outputs
    for output in torch.arange(prediction.shape[1]):
        time_deriv, du = library_deriv(data, prediction[:, output:output+1], library_config)
        u = library_poly(prediction[:, output:output+1], library_config)

        poly_list.append(u)
        deriv_list.append(du)
        time_deriv_list.append(time_deriv)

    samples = time_deriv_list[0].shape[0]
    total_terms = poly_list[0].shape[1] * deriv_list[0].shape[1]

    # Calculating theta
    if len(poly_list) == 1:
        theta = torch.matmul(poly_list[0][:, :, None], deriv_list[0][:, None, :]).view(samples, total_terms) # If we have a single output, we simply calculate and flatten matrix product between polynomials and derivatives to get library
    else:
        theta_uv = torch.cat([torch.matmul(u[:, :, None], v[:, None, :]).view(samples, total_terms) for u, v in combinations(poly_list, 2)], 1)  # calculate all unique combinations between polynomials
        theta_dudv = torch.cat([torch.matmul(du[:, :, None], dv[:, None, :]).view(samples, total_terms)[:, 1:] for du, dv in combinations(deriv_list, 2)], 1) # calculate all unique combinations of derivatives
        theta_udu = torch.cat([torch.matmul(u[:, 1:, None], du[:, None, 1:]).view(samples, (poly_list[0].shape[1]-1) * (deriv_list[0].shape[1]-1)) for u, dv in product(poly_list, deriv_list)], 1)  # calculate all unique products of polynomials and derivatives
        theta = torch.cat([theta_uv, theta_dudv, theta_udu], dim=1)

    return time_deriv_list, theta



