"""Fast execution check using all selected rows, with only two epochs."""
from copy import deepcopy
from src.configs.vowels import CONFIG as BASE

CONFIG = deepcopy(BASE)
CONFIG.training.epochs = 2
CONFIG.training.pretrain_epochs = 1
CONFIG.output_dir = "outputs/smoke"
