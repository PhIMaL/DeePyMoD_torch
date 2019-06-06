import numpy as np 
import torch, torch.nn
from torch.autograd import grad

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

    return y_t, theta

def library_kinetic(data, prediction,library_config):
    
    u = prediction[:,0:1] 
    v = prediction[:,1:2]
    du = grad(u, data, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_t = du[:, 0:1]
    
    dv = grad(v, data, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_t = dv[:, 0:1]
    
    theta = torch.cat([u, v], dim=1)
    
    return  torch.cat([u_t,v_t], dim=1), theta 