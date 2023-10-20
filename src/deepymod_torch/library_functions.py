import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations, product
from functools import reduce

def library_poly(prediction, max_order):
    # Calculate the polynomials of u (technically these are monomials)
    u = torch.ones_like(prediction)
    for order in np.arange(1, max_order+1):
        u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

    return u


def library_deriv(data, prediction, max_order):
    """
    Computes the time derivative and up to max_order spatial derivatives of the prediction tensor with respect to the data tensor.
    
    Args:
        data (torch.Tensor): Input tensor of shape (batch_size, input_dim). Example: X_train.
        prediction (torch.Tensor): Output tensor of shape (batch_size, output_dim). Example: y_train_pred.
        max_order (int): Maximum order of spatial derivatives to compute.
        
    Returns:
        time_deriv (torch.Tensor): Time derivative of the prediction tensor with respect to the data tensor.
        du (torch.Tensor): Tensor of shape (batch_size, max_order+1) containing the computed spatial derivatives.
    """
    
    dy = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0] # Calculate first order derivatives of prediction with respect to data
    time_deriv = dy[:, 0:1] # First column is time derivative
    
    if max_order == 0: # If we only want the time derivative, du is just a scalar
        du = torch.ones_like(time_deriv)
    else: # Else we calculate the spatial derivatives
        du = torch.cat((torch.ones_like(time_deriv), dy[:, 1:2]), dim=1) # second column of dy gives first order derivative
        if max_order > 1: # If we want higher order derivatives, we calculate them successively and concatenate them to du
            for order in np.arange(1, max_order):
                du = torch.cat((du, grad(du[:, order:order+1], data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 1:2]), dim=1)
    
    return time_deriv, du


def library_1D_in(input, poly_order, diff_order):
    """
    Computes the library matrix for a spatial 1D input signal, given the input data, the maximum polynomial order and the maximum derivative order.
    
    Parameters
    ----------
    input : tuple of two torch.Tensor
        A tuple containing the prediction tensor and the data tensor, both of shape (samples, features).
    poly_order : int
        The maximum polynomial order to include in the library.
    diff_order : int
        The maximum derivative order to include in the library.
    
    Returns
    -------
    time_deriv_list : list of torch.Tensor
        A list containing the time derivative tensors for each output feature, each of shape (samples, 1).
    theta : torch.Tensor
        The library matrix, of shape (samples, total_terms), where total_terms is the total number of terms in the library.
        when poly_order=2 and diff_order=3 and we have a single output the theta matrix has columns:
            ['', 'u_x', 'u_xx', 'u_xxx', 'u', 'uu_x', 'uu_xx', 'uu_xxx', 'u^2', 'u^2u_x', 'u^2u_xx', 'u^2u_xxx']
        For more details run utilities.terms_definition()
    """
    prediction, data = input
    poly_list = []
    deriv_list = []
    time_deriv_list = []

    # Creating lists for all outputs 
    for output in torch.arange(prediction.shape[1]): # loop over all dynamical fields modelled by PDE (in case we have system of PDEs, i.e. more than one dynamical field)
        time_deriv, du = library_deriv(data, prediction[:, output:output+1], diff_order)
        u = library_poly(prediction[:, output:output+1], poly_order)

        poly_list.append(u)
        deriv_list.append(du)
        time_deriv_list.append(time_deriv)

    samples = time_deriv_list[0].shape[0] # number of samples
    total_terms = poly_list[0].shape[1] * deriv_list[0].shape[1] # product of the number of possible polynomials (i.e. monomials) and the number of derivative terms
    
    # Calculating theta matrix (equation (4) of the manuscript)
    if len(poly_list) == 1:  # If we have a single output (one dynamical field modelled by the PDE), we simply calculate and flatten matrix product between polynomials and derivatives to get library
        theta = torch.matmul(poly_list[0][:, :, None], deriv_list[0][:, None, :]).view(samples, total_terms) 
        # For each sample poly_list[0][each_sample, :] and deriv_list[0][each_sample, :] the above line is equivalent to np.multiply.outer(poly_list[0][each_sample, :],deriv_list[0][each_sample, :] ).reshape(-1)
        # so the logic of the expression can be understood by executing np.add.outer(np.array(['', 'u', 'u^2'], object),np.array(['', 'u_x', 'u_xx','u_xxx'], object)).reshape(-1) <- this is consistent with equation (4)
        # this means that we iterate over deriv_list first (fast index) and then over poly_list (slow index)
        # this gives, for example: ['', 'u_x', 'u_xx', 'u_xxx', 'u', 'uu_x', 'uu_xx', 'uu_xxx', 'u^2', 'u^2u_x', 'u^2u_xx', 'u^2u_xxx']
    else:
        theta_uv = reduce((lambda x, y: (x[:, :, None] @ y[:, None, :]).view(samples, -1)), poly_list)  # TODO comment the following lines
        theta_dudv = torch.cat([torch.matmul(du[:, :, None], dv[:, None, :]).view(samples, -1)[:, 1:] for du, dv in combinations(deriv_list, 2)], 1) # calculate all unique combinations of derivatives
        theta_udu = torch.cat([torch.matmul(u[:, 1:, None], du[:, None, 1:]).view(samples, (poly_list[0].shape[1]-1) * (deriv_list[0].shape[1]-1)) for u, dv in product(poly_list, deriv_list)], 1)  # calculate all unique products of polynomials and derivatives
        theta = torch.cat([theta_uv, theta_dudv, theta_udu], dim=1)
        
    return time_deriv_list, theta


def library_2Din_1Dout(input, poly_order, diff_order):
        '''
        Constructs a library graph in 1D. Library config is dictionary with required terms.
        '''
        prediction, data = input
        # Polynomial
        
        u = torch.ones_like(prediction)
        for order in np.arange(1, poly_order+1):
            u = torch.cat((u, u[:, order-1:order] * prediction), dim=1)

        # Gradients
        du = grad(prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = grad(u_x, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = grad(u_y, data, grad_outputs=torch.ones_like(prediction), create_graph=True)[0][:, 2:3]
 
        du = torch.cat((torch.ones_like(u_x), u_x, u_y , u_xx, u_yy, u_xy), dim=1)

        samples= du.shape[0]
        # Bringing it together
        theta = torch.matmul(u[:, :, None], du[:, None, :]).view(samples,-1)
        
        return [u_t], theta
    