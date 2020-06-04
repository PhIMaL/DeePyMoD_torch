# General imports
import numpy as np
import torch

# DeepMoD stuff
from deepymod_torch import DeepMoD
from deepymod_torch.model.networks import NN
from deepymod_torch.model.library import library_1D_in, Library
from deepymod_torch.model.constraint import LstSq
from deepymod_torch.model.sparse_estimators import Clustering

from phimal_utilities.data import Dataset
from phimal_utilities.data.burgers import BurgersDelta

# Setting cuda
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Making data
v = 0.1
A = 1.0

# Making grid
x = np.linspace(-3, 4, 100)
t = np.linspace(0.5, 5.0, 50)
x_grid, t_grid = np.meshgrid(x, t, indexing='ij')
dataset = Dataset(BurgersDelta, v=v, A=A)

X_train, y_train = dataset.create_dataset(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), n_samples=1000, noise=1e-3)

# Configuring model
network = NN(2, [30, 30, 30, 30, 30], 1)
library = Library(library_1D_in, poly_order=1, diff_order=2)
estimator = Clustering()
constraint = LstSq()

model = DeepMoD(network, library, estimator, constraint)

# Running model
print(model(X_train))
