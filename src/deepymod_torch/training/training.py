import torch
import numpy as np
import time
from math import pi

from deepymod_torch.utils.tensorboard import Tensorboard
from deepymod_torch.utils.output import progress
from deepymod_torch.training.early_stopping import EarlyStopping


def train(model, data, target, optimizer, max_iterations=10000, stopper_kwargs={}, log_dir=None):
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms, log_dir)  # initializing custom tb board

    early_stopper = EarlyStopping(**stopper_kwargs)
    l1_previous_mask = None
    converged = False

    # Training
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in torch.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, sparse_thetas, thetas, constraint_coeffs = model(data)
        l1_norm = torch.stack([torch.sum(torch.abs(coeff_vector)) for coeff_vector in constraint_coeffs])

        MSE = torch.mean((prediction - target)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2) for dt, theta, coeff_vector in zip(time_derivs, sparse_thetas, constraint_coeffs)])
        loss = torch.sum(2 * torch.log(2 * pi * MSE) + Reg / (MSE + 1e-5))  # 1e-5 for numerical stability

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ====================== Logging =======================
        # Write progress to command line and tensorboard
        if iteration % 25 == 0:
            progress(iteration, start_time, max_iterations, loss.item(), torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())

            # We pad the sparse vectors with zeros so they get written correctly)
            coeff_vectors_padded = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.detach().squeeze()) for mask, coeff_vector in zip(model.constraint.sparsity_masks, constraint_coeffs)]
            board.write(iteration, loss, MSE, Reg, l1_norm, coeff_vectors_padded, coeff_vectors_padded)

        # ================== Validation and sparsity =============
        # Updating sparsity and or convergence
        early_stopper(iteration, torch.sum(l1_norm), model, optimizer)
        if early_stopper.early_stop is True:
            # Reset early and model
            early_stopper.reset()
            if early_stopper.first_sparsity_epoch == 1e8:  # if first time, reset to optimal model
                model.load_state_dict(torch.load('model_checkpoint.pt'))
                optimizer.load_state_dict(torch.load('optimizer_checkpoint.pt'))

                # Forward pass to get values at that point
                print('Updating mask.')
                prediction, time_derivs, sparse_thetas, thetas, constraint_coeffs = model(data)
                l1_norm = torch.stack([torch.sum(torch.abs(coeff_vector)) for coeff_vector in constraint_coeffs])

            with torch.no_grad():
                new_masks = model.calculate_sparsity_mask(thetas, time_derivs)
                masks_similar = np.all([torch.equal(new_mask, old_mask) for new_mask, old_mask in zip(new_masks, model.constraint.sparsity_masks)])
                model.constraint.sparsity_masks = new_masks
                print('\n', model.constraint.sparsity_masks)

                # Convergence when l1 norms of two subsequent masks are similar
                if l1_previous_mask is None:
                    l1_previous_mask = torch.sum(l1_norm)
                    converged = False
                elif (torch.abs(torch.sum(l1_norm) - l1_previous_mask) / l1_previous_mask < 0.05) and (masks_similar is True):
                    converged = True
                else:
                    l1_previous_mask = torch.sum(l1_norm)
                    converged = False

        # ================== Breaking loop if sparsity converged ==================
        if converged:
            print('Sparsity converged. Stopping training.')
            break

    board.close()
