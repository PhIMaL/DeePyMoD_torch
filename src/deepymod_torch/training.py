import torch
import time

from deepymod_torch.output import Tensorboard, progress
from deepymod_torch.losses import reg_loss, mse_loss, l1_loss
from deepymod_torch.sparsity import scaling

def train(data, target, model, optimizer, max_iterations, loss_func_args):
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model[-1].coeff_vector_list]
    board = Tensorboard(number_of_terms)

    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(0, max_iterations + 1):
        # Calculating prediction and library and scaling
        prediction, time_deriv_list, sparse_theta_list, coeff_vector_list = model(data)
        coeff_vector_scaled_list = [scaling(coeff_vector, sparse_theta, time_deriv) for time_deriv, sparse_theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, coeff_vector_list)]

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

