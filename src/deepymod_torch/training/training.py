import torch
import time
from math import pi
import numpy as np

from ..utils.tensorboard import Tensorboard
from ..utils.output import progress
from .convergence import Convergence
from ..model.deepmod import DeepMoD
from typing import Optional


def train(model: DeepMoD,
          data: torch.Tensor,
          target: torch.Tensor,
          optimizer,
          sparsity_scheduler,
          log_dir: Optional[str] = None,
          max_iterations: int = 10000,
          write_iterations: int = 25,
          **convergence_kwargs) -> None:
    """[summary]

    Args:
        model (DeepMoD): [description]
        data (torch.Tensor): [description]
        target (torch.Tensor): [description]
        optimizer ([type]): [description]
        sparsity_scheduler ([type]): [description]
        log_dir (Optional[str], optional): [description]. Defaults to None.
        max_iterations (int, optional): [description]. Defaults to 10000.
    """
    start_time = time.time()
    board = Tensorboard(log_dir)  # initializing tb board

    # Training
    convergence = Convergence(**convergence_kwargs)
    print('| Iteration | Progress | Time remaining |     Loss |      MSE |      Reg |    L1 norm |')
    for iteration in np.arange(0, max_iterations + 1):
        # ================== Training Model ============================
        prediction, time_derivs, thetas = model(data)

        MSE = torch.mean((prediction - target)**2, dim=0)  # loss per output
        Reg = torch.stack([torch.mean((dt - theta @ coeff_vector)**2)
                           for dt, theta, coeff_vector in zip(time_derivs, thetas, model.constraint_coeffs(scaled=False, sparse=True))])
        loss = torch.sum(2 * torch.log(2 * pi * MSE) + Reg / (MSE + 1e-6))  # 1e-5 for numerical stability

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ====================== Logging =======================
        # We calculate the normalization factor and the l1_norm
        l1_norm = torch.sum(torch.abs(torch.cat(model.constraint_coeffs(sparse=True, scaled=True), dim=1)), dim=0)

        # Write progress to command line and tensorboard
        if iteration % write_iterations == 0:
            progress(iteration, start_time, max_iterations, loss.item(),
                     torch.sum(MSE).item(), torch.sum(Reg).item(), torch.sum(l1_norm).item())

            if model.estimator_coeffs() is None:
                estimator_coeff_vectors = [torch.zeros_like(coeff) for coeff in model.constraint_coeffs(sparse=True, scaled=False)] # It doesnt exist before we start sparsity, so we use zeros
            else:
                estimator_coeff_vectors = model.estimator_coeffs()

            board.write(iteration, loss, MSE, Reg, l1_norm, model.constraint_coeffs(sparse=True, scaled=True), model.constraint_coeffs(sparse=True, scaled=False), estimator_coeff_vectors)
            
        # ================== Validation and sparsity =============
        # Updating sparsity and or convergence
        sparsity_scheduler(iteration, torch.sum(l1_norm))
        if sparsity_scheduler.apply_sparsity is True:
            with torch.no_grad():
                model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivs)
                sparsity_scheduler.reset()
                print(model.sparsity_masks)

        # Checking convergence
        convergence(iteration, torch.sum(l1_norm))
        if convergence.converged is True:
            print('Algorithm converged. Stopping training.')
            break
    board.close()