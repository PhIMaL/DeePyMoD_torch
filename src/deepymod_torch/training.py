import torch
import time, sys
from torch.utils.tensorboard import SummaryWriter

from deepymod_torch.sparsity import scaling
from deepymod_torch.tensorboard import custom_board


def train_mse():
    return 'not implemented yet'


def train_cycle(data, target, model, optimizer, max_iterations, l1):
    start_time = time.time()

    # preparing tensorboard writer
    writer = SummaryWriter()
    writer.add_custom_scalars(custom_board(model.coeff_vector_list))

    # Training
    print('| Iteration | Progress | Time remaining |     Cost |      MSE |      Reg |       L1 |')
    for iteration in torch.arange(1, max_iterations + 1):
        # Calculating prediction and library
        prediction, time_deriv_list, theta = model(data)
        sparse_theta_list = [theta[:, sparsity_mask] for sparsity_mask in model.sparsity_mask_list]

        # Scaling
        coeff_vector_scaled_list = [scaling(coeff_vector, sparse_theta, time_deriv) for time_deriv, sparse_theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, model.coeff_vector_list)]

        # Calculating PI
        reg_cost_list = torch.stack([torch.mean((time_deriv - sparse_theta @ coeff_vector)**2) for time_deriv, sparse_theta, coeff_vector in zip(time_deriv_list, sparse_theta_list, model.coeff_vector_list)])
        loss_reg = torch.sum(reg_cost_list)

        # Calculating MSE
        MSE_cost_list = torch.mean((prediction[0] - target)**2, dim=0)
        loss_MSE = torch.sum(MSE_cost_list)

        # Calculating L1
        l1_cost_list = torch.stack([torch.sum(torch.abs(coeff_vector_scaled)) for coeff_vector_scaled in coeff_vector_scaled_list])
        loss_l1 = l1 * torch.sum(l1_cost_list)

        # Calculating total loss
        loss = loss_MSE + loss_reg + loss_l1

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 50 == 0:
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

            # Printing
            elapsed_time = time.time() - start_time
            progress(iteration, max_iterations, elapsed_time, loss.item(), loss_MSE.item(), loss_reg.item(), loss_l1.item())
    writer.close()

def progress(iteration, max_iteration, elapsed_time, cost, MSE, PI, L1):
    percent = iteration.item()/max_iteration * 100
    time_left = elapsed_time * (max_iteration/iteration - 1) if iteration != 0 else 0
    sys.stdout.write(f"\r  {iteration:>9}   {percent:>7.2f}%   {time_left:>13.0f}s   {cost:>8.2e}   {MSE:>8.2e}   {PI:>8.2e}   {L1:>8.2e} ")
    sys.stdout.flush()