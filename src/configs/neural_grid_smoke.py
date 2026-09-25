"""Exercise both search stages, checkpoint reload, selection and held-out scoring."""
from copy import deepcopy
from src.configs.neural_grid import SEARCH as BASE

SEARCH = deepcopy(BASE)
SEARCH.grid = {"hidden_dims": [(16,)], "activation": ["tanh"],
               "learning_rate": [1e-3], "batch_size": [1024]}
SEARCH.base.training.pretrain_epochs = 1
SEARCH.screening_epochs = 1
SEARCH.confirmation_epochs = 1
SEARCH.finalists = 1
SEARCH.confirmation_seeds = (42,)
SEARCH.output_dir = "outputs/neural_grid_smoke"
