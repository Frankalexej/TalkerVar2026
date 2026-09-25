"""Same data/training defaults as vowels.py; add one shared auxiliary w."""
from copy import deepcopy
from dataclasses import asdict

from src.config import SharedWModelConfig
from src.configs.vowels import CONFIG as BASE

CONFIG = deepcopy(BASE)
CONFIG.model = SharedWModelConfig(
    **asdict(BASE.model),
    w_dim=2,
    prior_hidden_dims=(64,),
    objective="paper",  # Paper Eq. 5 estimator. Set "structured" for exact ELBO MC.
)
CONFIG.methods = ("gmvae_shared_w",)
CONFIG.output_dir = "outputs/shared_w"
