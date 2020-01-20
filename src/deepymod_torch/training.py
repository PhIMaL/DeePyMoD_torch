import torch
import time, sys
from torch.utils.tensorboard import SummaryWriter

from deepymod_torch.sparsity import scaling
from deepymod_torch.tensorboard import custom_board
from deepymod_torch.losses import RegLoss, MSELoss, L1Loss, DeepMoDLoss


def train(data, target, model, optimizer, max_iterations, loss_func=DeepMoDLoss, loss_func_args={'l1': 10e-5}):
    start_time = time.time()
    loss_func = loss_func(**loss_func_args)
    
    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(1, max_iterations + 1):
        # Calculating prediction and library and scaling
        output = model(data)  # prediction, time_deriv_list, theta
        loss = loss_func(output, target, model.coeff_vector_list)
        
        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 50 == 0:
            elapsed_time = time.time() - start_time
            progress(iteration, max_iterations, elapsed_time, loss.item(), 0, 0, 0)


def progress(iteration, max_iteration, elapsed_time, cost, MSE, PI, L1):
    percent = iteration.item()/max_iteration * 100
    time_left = elapsed_time * (max_iteration/iteration - 1) if iteration != 0 else 0
    sys.stdout.write(f"\r  {iteration:>9}   {percent:>7.2f}%   {time_left:>13.0f}s   {cost:>8.2e}   {MSE:>8.2e}   {PI:>8.2e}   {L1:>8.2e} ")
    sys.stdout.flush()

'''
    # preparing tensorboard writer
    writer = SummaryWriter()
    writer.add_custom_scalars(custom_board(model.coeff_vector_list))
# Tensorboard stuff
            writer.add_scalar('Total loss', loss, iteration)
            for idx in torch.arange(len(MSE_cost_list)):
                # Costs
                writer.add_scalar('MSE '+str(idx), MSE_cost_list[idx], iteration)
                writer.add_scalar('Regression '+str(idx), reg_cost_list[idx], iteration)
                writer.add_scalar('L1 '+str(idx), l1_cost_list[idx], iteration)

                # Coefficients
                for element_idx, element in enumerate(torch.unbind(model.coeff_vector_list[idx])):
                    writer.add_scalar('coeff ' + str(idx) + ' ' + str(element_idx), element, iteration)

                # Scaled coefficients
                for element_idx, element in enumerate(torch.unbind(coeff_vector_scaled_list[idx])):
                    writer.add_scalar('scaled_coeff ' + str(idx) + ' ' + str(element_idx), element, iteration)

    writer.close()
'''