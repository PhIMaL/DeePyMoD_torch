import numpy as np
import torch.nn as nn
import torch

from library_function import *

def LinNetwork(network_config): # network config
    
    input_dim=network_config['input_dim'] 
    hidden_dim=network_config['hidden_dim'] 
    layers=network_config['layers']
    output_dim=network_config['output_dim']
    
    modules = []
    modules.append(nn.Linear(input_dim, hidden_dim))
    modules.append(nn.Tanh())
    for i in np.arange(0,layers):
        modules.append(nn.Linear(hidden_dim, hidden_dim))
        modules.append(nn.Tanh())
    modules.append(nn.Linear(hidden_dim, output_dim))
    sequential = nn.Sequential(*modules)
    
    return sequential

def Training(data, target, optim_config, library_config, network, network_config):
    
    max_it = optim_config['max_iteration']
    l1 = optim_config['lambda']
    
    # Initialize the weight vector and optimizer 
    weight_vector = torch.ones((library_config['total_terms'], network_config['output_dim']), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([{'params':network.parameters()}, {'params': weight_vector}])
    
    for iteration in np.arange(max_it):
        
        # Calculate the predicted y-value, construct the library function
        prediction = network(data)    
        y_t, theta = library_poly_multi(data, prediction,library_config)
        f = y_t - theta @ weight_vector

        # Losses: MSE, PI and L1  

        loss_MSE = torch.nn.MSELoss()(prediction, target)
        loss_PI = torch.nn.MSELoss()(f, torch.zeros_like(y_t))
        loss_L1 = l1*nn.L1Loss()(weight_vector,torch.zeros_like(weight_vector))
        
        # Combine all the losses
        loss = loss_MSE + loss_PI + loss_L1 
        
        # Optimizwe step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the losses during training 
        if iteration == 0:
            print('Epoch | Total loss | MSE | PI | L1 ')
        if iteration % 1000 == 0:
            print(iteration, "%.1E" % loss.detach().numpy(), "%.1E" % loss_MSE.detach().numpy(), "%.1E" % loss_PI.detach().numpy(), "%.1E" % loss_L1.detach().numpy())
            print(np.around(weight_vector.detach().numpy(),decimals=2))
            
    return y_t, theta, weight_vector


def Final_Training(data, target, optim_config, library_config, network, network_config, sparse_weight_vector, sparsity_mask):
    
    max_it = 5000
    
    # Re-initialize the optimizer 
    optimizer = torch.optim.Adam([{'params':network.parameters()}, {'params': sparse_weight_vector}])
    
    for iteration in np.arange(max_it):
        
        # Calculate the predicted y-value, construct the sparse library function
        prediction = network(data)    
        y_t, theta = library_poly_multi(data, prediction,library_config)

        dummy = torch.zeros((library_config['total_terms']*network_config['output_dim'],1))
        dummy[sparsity_mask] = sparse_weight_vector
        dummy = dummy.reshape(library_config['total_terms'], network_config['output_dim'])
        f = y_t - theta @ dummy
        
        # Losses: MSE, PI and L1  
        loss_MSE = torch.nn.MSELoss()(prediction, target)
        loss_PI = torch.nn.MSELoss()(f, torch.zeros_like(y_t))
        
        # Combine all the losses
        loss = loss_MSE + loss_PI
        
        # Optimizwe step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the losses during training 
        if iteration == 0:
            print('Epoch | Total loss | MSE | PI ')
        if iteration % 1000 == 0:
            print(iteration, "%.1Ef" % loss.detach().numpy(), "%.1E" % loss_MSE.detach().numpy(), "%.1E" % loss_PI.detach().numpy())
            print(np.around(sparse_weight_vector.detach().numpy(),decimals=2).reshape(1,-1))
    return sparse_weight_vector, prediction

def Training_MSE(data, target, optim_config, network, network_config):
    
    max_it = optim_config['max_iteration']
    l1 = optim_config['lambda']
    
    # Initialize the weight vector and optimizer 

    optimizer = torch.optim.Adam([{'params':network.parameters()}])
    
    for iteration in np.arange(max_it):
        
        prediction = network(data)    
        # Combine all the losses
        loss = torch.nn.MSELoss()(prediction, target)
        
        # Optimizwe step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the losses during training 
        if iteration == 0:
            print('Epoch | Total loss | MSE ')
        if iteration % 1000 == 0:
            print(iteration, "%.1E" % loss.detach().numpy(), "%.1E" % loss.detach().numpy())
            
    return prediction, network


def Training_PI(data, target, optim_config, library_config, network, network_config,init_coeff):
    
    max_it = optim_config['max_iteration']
    l1 = optim_config['lambda']
    
    # Initialize the weight vector and optimizer 
    weight_vector = init_coeff
    optimizer = torch.optim.Adam([{'params': weight_vector}])
    
    for iteration in np.arange(max_it):
        
        # Calculate the predicted y-value, construct the library function
        prediction = network(data)    
        y_t, theta = library_poly_multi(data, prediction,library_config)
        f = y_t - theta @ weight_vector

        # Losses: MSE, PI and L1  
        loss_MSE = torch.nn.MSELoss()(prediction, target)
        loss_PI = torch.nn.MSELoss()(f, torch.zeros_like(y_t))
        loss_L1 = l1*nn.L1Loss()(weight_vector,torch.zeros_like(weight_vector))
        
        # Combine all the losses
        loss = loss_MSE + loss_PI + loss_L1 
        
        # Optimizwe step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the losses during training 
        if iteration == 0:
            print('Epoch | Total loss | MSE | PI | L1 ')
        if iteration % 1000 == 0:
            print(iteration, "%.1E" % loss.detach().numpy(), "%.1E" % loss_MSE.detach().numpy(), "%.1E" % loss_PI.detach().numpy(), "%.1E" % loss_L1.detach().numpy())
            print(np.around(weight_vector.detach().numpy(),decimals=2))
            
    return y_t, theta, weight_vector