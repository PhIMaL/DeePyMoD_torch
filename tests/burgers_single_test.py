# Diffusion, D = 0.5, a = 0.25

# General imports
import numpy as np
import torch

# DeepMoD stuff
from deepymod_torch.library_function import library_new
from deepymod_torch.neural_net import train, deepmod_init
from deepymod_torch.sparsity import scaling, threshold
from deepymod_torch.nn import create_deriv_data

# Setting cuda
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Loading data
data = np.load('../data/processed/burgers.npy', allow_pickle=True).item()
X = np.transpose((data['t'].flatten(), data['x'].flatten()))
y = np.real(data['u']).reshape((data['u'].size, 1))
number_of_samples = 1000

idx = np.random.permutation(y.size)
X_train = torch.tensor(X[idx, :][:number_of_samples], dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y[idx, :][:number_of_samples], dtype=torch.float32)

optim_config = {'lambda': 10**-5, 'max_iterations': }
library_config = {'type': library_new, 'poly_order': 2, 'diff_order': 2}
network_config = {'input_dim': 2, 'hidden_dim': 20, 'layers': 5, 'output_dim': 1}
X_train = create_deriv_data(X_train, 3)

network, coeff_vector_list, sparsity_mask_list = deepmod_init(network_config, library_config)

network