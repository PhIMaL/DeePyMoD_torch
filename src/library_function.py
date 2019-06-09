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
    
    theta = torch.cat([torch.ones_like(u), u, v], dim=1)
    print(torch.cat((u_t,v_t), dim=1))
    return  torch.cat((u_t,v_t), dim=1), theta 

def library_1D_2Dout(data, prediction,library_config):
   
    max_order = library_config['poly_order']
    max_diff = library_config['diff_order']
    
    # Calculate the polynomes of u 
    
    y = prediction[:,0:1] 
    u = torch.ones_like(prediction[:,0:1])
    for order in np.arange(1,max_order+1):
        u = torch.cat((u, u[:, order-1:order]*prediction[:,0:1]),dim=1)
    
    v = torch.ones_like(prediction[:,1:2])
    for order in np.arange(1,max_order+1):
        v = torch.cat((v, v[:, order-1:order]*prediction[:,1:2]),dim=1)
    print('u shape', u.shape)
    # Calculate the derivatives
    
    dU = grad(prediction[:,0:1], data, grad_outputs=torch.ones_like(prediction[:,0:1]), create_graph=True)[0]
    u_x = dU[:, 0:1]
    u_t = dU[:, 1:2]
    dU2 = grad(u_x, data, grad_outputs=torch.ones_like(prediction[:,0:1]), create_graph=True)[0]
    u_xx = dU2[:,0:1]
    du = torch.cat((u_x.reshape(-1,1), u_xx.reshape(-1,1)),dim=1).unsqueeze(2)

    
    dV = grad(prediction[:,1:2], data, grad_outputs=torch.ones_like(prediction[:,1:2]), create_graph=True)[0]
    v_x = dV[:, 0:1]
    v_t = dV[:, 1:2]
    dV2 = grad(v_x, data, grad_outputs=torch.ones_like(prediction[:,0:1]), create_graph=True)[0]
    v_xx = dV2[:,0:1]
    
    dv = torch.cat((v_x.reshape(-1,1), v_xx.reshape(-1,1)),dim=1).unsqueeze(1)

    
    Ddudv = (du @ dv)
    Ddudv = torch.reshape(Ddudv,(Ddudv.shape[0],-1))
    
    dudv =  torch.cat([torch.ones_like(prediction[:,0:1]), du.squeeze(), dv.squeeze(), Ddudv], dim=1).unsqueeze(1)
    
    
    uv = torch.matmul(u.unsqueeze(2),v.unsqueeze(1))
    uv = torch.reshape(uv,(uv.shape[0],-1)).unsqueeze(2)
    # Constructing Theta
    theta = uv @ dudv

    theta = torch.reshape(theta,(theta.shape[0],-1))

    return torch.cat((u_t,v_t), dim=1), theta

def library_poly(data, prediction,library_config):
   
    max_order = library_config['poly_order']
    
    # Calculate the polynomes of u 
    
    y = prediction[:,0:1] 
    u = torch.ones_like(prediction[:,0:1])
    for order in np.arange(1,max_order+1):
        u = torch.cat((u, u[:, order-1:order]*prediction[:,0:1]),dim=1)
        
    v = torch.ones_like(prediction[:,1:2])
    for order in np.arange(1,max_order+1):
        v = torch.cat((v, v[:, order-1:order]*prediction[:,1:2]),dim=1)
    # Calculate the derivatives
    
    dU = grad(prediction[:,0:1], data, grad_outputs=torch.ones_like(prediction[:,0:1]), create_graph=True)[0]
    u_t = dU[:, 0:1]
    
    dV = grad(prediction[:,1:2], data, grad_outputs=torch.ones_like(prediction[:,1:2]), create_graph=True)[0]
    v_t = dV[:, 0:1]    
    
    uv = torch.matmul(u.unsqueeze(2),v.unsqueeze(1))
    theta = torch.reshape(uv,(uv.shape[0],-1)).unsqueeze(2)
    # Constructing Theta
    
    theta = torch.reshape(theta,(theta.shape[0],-1))

    return torch.cat((u_t,v_t), dim=1), theta

def library_poly_multi(data, prediction,library_config):
   
    max_order = library_config['poly_order']
    
    # Calculate the polynomes of u 
    
    y = prediction[:,0:1] 
    u = torch.ones_like(prediction[:,0:1])
    for order in np.arange(1,max_order+1):
        u = torch.cat((u, u[:, order-1:order]*prediction[:,0:1]),dim=1)
        
    v = torch.ones_like(prediction[:,1:2])
    for order in np.arange(1,max_order+1):
        v = torch.cat((v, v[:, order-1:order]*prediction[:,1:2]),dim=1)
    # Calculate the derivatives
    
    dU = grad(prediction[:,0:1], data, grad_outputs=torch.ones_like(prediction[:,0:1]), create_graph=True)[0]
    u_t = dU[:, 0:1]
    u_x = dU[:,1:2]
    
    dV = grad(prediction[:,1:2], data, grad_outputs=torch.ones_like(prediction[:,1:2]), create_graph=True)[0]
    v_t = dV[:, 0:1]    
    v_x = dV[:,1:2]
    
    
    uv = torch.matmul(u.unsqueeze(2),v.unsqueeze(1))
    uv = torch.reshape(uv,(uv.shape[0],-1))
 #   uv = torch.cat((uv,u_x,v_x),dim=1)
    
    theta = uv

    return torch.cat((u_t,v_t), dim=1), theta
