from deepymod_torch.deepmod import DeepMoD


class DeepMoDLi(pl.LightningModule, DeepMoD):
    def __init__(self, function_approximator, library, constraint, sparsity_estimator):
        super(DeepMoD, self).__init__(function_approximator, library, constraint, sparsity_estimator)


    def training_step(self, batch, batch_idx):
        data, target = batch
        prediction, time_derivs, sparse_thetas, thetas, constraint_coeff_vectors = self(data)

        MSE = torch.mean((prediction - target)**2, dim=0)
        Reg = torch.stack([torch.mean((time_deriv - theta @ coeff_vector)**2)
                          for time_deriv, theta, coeff_vector in zip(time_derivs, sparse_thetas, constraint_coeff_vectors)])

        loss = torch.sum(2 * torch.log(2 * pi * MSE) + Reg / MSE)

        return {'loss': loss}
s
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), betas=(0.99, 0.999), amsgrad=True)
