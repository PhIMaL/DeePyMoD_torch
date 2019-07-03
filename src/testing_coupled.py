import sys
sys.path.append('src/')
import numpy as np
import torch
from library_function import library_1D
from DeepMod import DeepMoD
from scipy.io import loadmat

np.random.seed(42)
number_of_samples = 1000

raw = loadmat('data/Keller_Segel.mat')['Expression1']
x = raw[0::4]
t = raw[1::4]
usol = raw[2::4]
vsol = raw[3::4]

X = np.concatenate((x, t), axis=1)
y = np.concatenate((usol, vsol), axis=1)

idx = np.random.permutation(y.shape[0])
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y[idx, :][:number_of_samples], dtype=torch.float32)

optim_config = {'lambda': 10**-6, 'max_iterations': 10000}
lib_config = {'type': library_1D, 'poly_order': 2, 'diff_order': 3, 'total_terms': 12}
network_config = {'input_dim': 2, 'hidden_dim': 20, 'layers': 5, 'output_dim': 1}

sparse_coeff_vector, sparsity_mask, network = DeepMoD(X_train, y_train, network_config, lib_config, optim_config)

print(sparse_coeff_vector)
