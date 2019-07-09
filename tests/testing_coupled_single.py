import numpy as np
import torch
from deepymod_torch.library_function import library_1D_in, library_1D
from deepymod_torch.DeepMod import DeepMoD

torch.set_default_tensor_type('torch.cuda.FloatTensor')  # enable for gpu.

np.random.seed(42)
number_of_samples = 1000

data = np.load('data/processed/burgers.npy', allow_pickle=True).item()

X = np.transpose((data['t'].flatten(), data['x'].flatten()))
y = np.real(data['u']).reshape((data['u'].size, 1))

idx = np.random.permutation(y.size)
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y[idx, :][:number_of_samples], dtype=torch.float32)

optim_config = {'lambda': 10**-6, 'max_iterations': 15000}
lib_config = {'type': library_1D_in, 'poly_order': 2, 'diff_order': 3}
network_config = {'input_dim': 2, 'hidden_dim': 20, 'layers': 5, 'output_dim': 1}


sparse_coeff_vector, sparsity_mask, network = DeepMoD(X_train, y_train, network_config, lib_config, optim_config)

print(sparse_coeff_vector, sparsity_mask)
