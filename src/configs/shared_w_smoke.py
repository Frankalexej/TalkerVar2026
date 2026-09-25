"""Full selected dataset, but only two epochs for an execution check."""
from copy import deepcopy
from src.configs.shared_w import CONFIG as BASE

CONFIG = deepcopy(BASE)
CONFIG.training.pretrain_epochs = 1
CONFIG.training.epochs = 2
CONFIG.output_dir = "outputs/shared_w_smoke"
