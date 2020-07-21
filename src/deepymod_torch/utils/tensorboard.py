from torch.utils.tensorboard import SummaryWriter


class Tensorboard():
    '''Tensorboard class for logging during deepmod training. '''
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        #self.writer.add_custom_scalars(custom_board(number_of_terms))

    def write(self, iteration, loss, loss_mse, loss_reg, loss_l1,
              constraint_coeff_vectors, unscaled_constraint_coeff_vectors, estimator_coeff_vectors, **kwargs):
        # Costs and coeff vectors
        self.writer.add_scalar('Loss/Loss', loss, iteration)
        self.writer.add_scalars('Loss/MSE', {f'Output_{idx}': val for idx, val in enumerate(loss_mse)}, iteration)
        self.writer.add_scalars('Loss/Reg', {f'Output_{idx}': val for idx, val in enumerate(loss_reg)}, iteration)
        self.writer.add_scalars('Loss/L1', {f'Output_{idx}': val for idx, val in enumerate(loss_l1)}, iteration)

        for output_idx, (coeffs, unscaled_coeffs, estimator_coeffs) in enumerate(zip(constraint_coeff_vectors, unscaled_constraint_coeff_vectors, estimator_coeff_vectors)):
            self.writer.add_scalars(f'Coeffs/Output_{output_idx}', {f'Coeff_{idx}': val for idx, val in enumerate(coeffs.squeeze())}, iteration)
            self.writer.add_scalars(f'Unscaled coeffs/Output_{output_idx}', {f'Coeff_{idx}': val for idx, val in enumerate(unscaled_coeffs.squeeze())}, iteration)
            self.writer.add_scalars(f'Estimator coeffs/Output_{output_idx}', {f'Coeff_{idx}': val for idx, val in enumerate(estimator_coeffs.squeeze())}, iteration)
        self.writer.close()
    
        # Writing remaining kwargs
        for key, value in kwargs.items():
            if value.numel() == 1:
                self.writer.add_scalar(f'Remaining/{key}', value, iteration)
            else:
                self.writer.add_scalars(f'Remaining/{key}', {f'val_{idx}': val.squeeze() for idx, val in enumerate(value.squeeze())}, iteration)
        
