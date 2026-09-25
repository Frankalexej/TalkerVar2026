"""24 Cartesian grid settings + original control; two finalists, three seeds."""
from copy import deepcopy
from src.configs.vowels import CONFIG as BASE
from src.search_config import GridSearchConfig

SEARCH = GridSearchConfig(
    base=deepcopy(BASE),
    grid={
        "hidden_dims": [(64,), (128, 128), (128, 128, 128)],
        "activation": ["relu", "tanh"],
        "learning_rate": [3e-4, 1e-3],
        "batch_size": [256, 1024],
    },
    screening_epochs=30,
    screening_patience=10,
    finalists=2,
    confirmation_epochs=100,
    confirmation_seeds=(42, 43, 44),
    workers=2,
    threads_per_worker=1,
)
