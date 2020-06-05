import torch
import time
from math import pi

from deepymod_torch.utils.tensorboard import Tensorboard
from deepymod_torch.utils.output import progress
from deepymod_torch.training.convergence import Convergence

def train(model, data, target, optimizer, sparsity_scheduler, log_dir=None, max_iterations=10000, **convergence_kwargs):
    start_time = time.time()
    number_of_terms = [coeff_vec.shape[0] for coeff_vec in model(data)[3]]
    board = Tensorboard(number_of_terms, log_dir)  # initializing custom tb board

    # Training
    convergence = Convergence(**convergence_kwargs)
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in torch.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, sparse_thetas, thetas, constraint_coeffs = model(data)
        
        MSE = torch.mean((prediction - target)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, sparse_thetas, constraint_coeffs)])
        loss = torch.sum(2 * torch.log(2 * pi * MSE) + Reg / (MSE + 1e-5))  # 1e-5 for numerical stability

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ====================== Logging =======================
        # Write progress to command line and tensorboard
        l1_norm = torch.stack([torch.sum(torch.abs(coeff_vector)) for coeff_vector in constraint_coeffs])
        if iteration % 25 == 0:
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())

            # We pad the sparse vectors with zeros so they get written correctly)
            constraint_coeff_vectors = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.detach().squeeze())
                                        for mask, coeff_vector
                                        in zip(model.constraint.sparsity_masks, constraint_coeffs)]
            unscaled_constraint_coeff_vectors = [torch.zeros(mask.size()).masked_scatter_(mask, coeff_vector.detach().squeeze()) / model.library.norm.squeeze()
                                                 for mask, coeff_vector
                                                 in zip(model.constraint.sparsity_masks, constraint_coeffs)]

            board.write(iteration, loss, MSE, Reg, l1_norm, constraint_coeff_vectors, unscaled_constraint_coeff_vectors)

        # ================== Validation and sparsity =============
        # Updating sparsity and or convergence
        sparsity_scheduler(iteration, torch.sum(l1_norm))
        if sparsity_scheduler.apply_sparsity is True:
            with torch.no_grad():
                model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                sparsity_scheduler.reset()
                print(model.constraint.sparsity_masks)

        # Checking convergence
        convergence(iteration, torch.sum(l1_norm))
        if convergence.converged is True:
            print('Algorithm converged. Stopping training.')
            break

    board.close()
