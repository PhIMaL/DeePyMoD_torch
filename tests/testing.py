import sys
sys.path.append('src/')
import numpy as np
import torch
from library_function import library_1D
from DeepMod import DeepMoD

np.random.seed(42)
number_of_samples = 1000

data = np.load('data/burgers.npy', allow_pickle=True).item()

X = np.transpose((data['x'].flatten(), data['t'].flatten()))
y = np.real(data['u']).reshape((data['u'].size, 1))

idx = np.random.permutation(y.size)
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y[idx, :][:number_of_samples], dtype=torch.float32)

optim_config = {'lambda': 10**-6, 'max_iterations': 10000}
lib_config = {'type': library_1D, 'poly_order': 2, 'diff_order': 3, 'total_terms': 12}
network_config = {'input_dim': 2, 'hidden_dim': 20, 'layers': 5, 'output_dim': 1}

sparse_coeff_vector, sparsity_mask, network = DeepMoD(X_train, y_train, network_config, lib_config, optim_config)

print(sparse_coeff_vector)
