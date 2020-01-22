import torch
import time

from deepymod_torch.output import Tensorboard, progress
from deepymod_torch.losses import reg_loss, mse_loss, l1_loss
from deepymod_torch.sparsity import scaling, threshold

def train(data, target, model, optimizer, max_iterations, loss_func_args):
    '''Trains the deepmod model with MSE, regression and l1 cost function. Updates model in-place.'''
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model[-1].coeff_vector_list]
    board = Tensorboard(number_of_terms)

    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 
        
        # Calculating loss
        loss_reg = reg_loss(time_deriv_list, sparse_theta_list, coeff_vector_list)
        loss_mse = mse_loss(prediction[0], target)
        loss_l1 = l1_loss(coeff_vector_scaled_list, loss_func_args['l1'])
        loss = torch.sum(loss_reg) + torch.sum(loss_mse) + torch.sum(loss_l1)
        
        # Writing
        if iteration % 100 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), torch.sum(loss_reg).item(), torch.sum(loss_l1).item())
            board.write(iteration, loss, loss_mse, loss_reg, loss_l1, coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()

def train_mse(data, target, model, optimizer, max_iterations, loss_func_args):
    '''Trains the deepmod model only on the MSE. Updates model in-place.'''
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model[-1].coeff_vector_list]
    board = Tensorboard(number_of_terms)

    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
        coeff_vector_scaled_list = scaling(coeff_vector_list, sparse_theta_list, time_deriv_list) 

        # Calculating loss
        loss_mse = mse_loss(prediction[0], target)
        loss = torch.sum(loss_mse)

        # Writing
        if iteration % 100 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(loss_mse).item(), 0, 0)
            board.write(iteration, loss, loss_mse, [0], [0], coeff_vector_list, coeff_vector_scaled_list)

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    board.close()

def train_deepmod(data, target, model, optimizer, max_iterations, loss_func_args):
    '''Performs full deepmod cycle: trains model, thresholds and trains again for unbiased estimate. Updates model in-place.'''
    # Train first cycle and get prediction
    train(data, target, model, optimizer, max_iterations, loss_func_args)
    prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)

    # Threshold, set sparsity mask and coeff vector
    sparse_coeff_vector_list, sparsity_mask_list = threshold(coeff_vector_list, sparse_theta_list, time_deriv_list)
    model[-1].sparsity_mask_list = sparsity_mask_list
    model[-1].coeff_vector_list = torch.nn.ParameterList(sparse_coeff_vector_list)
    print()
    print(sparse_coeff_vector_list)
    print(sparsity_mask_list)
    #Resetting optimizer for different shapes, train without l1 
    optimizer.param_groups[0]['params'] = model.parameters()
    print() #empty line for correct printing
    train(data, target, model, optimizer, max_iterations, dict(loss_func_args, **{'l1':0.0}))

