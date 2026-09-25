"""Configuration and strict scope boundaries for neural-only grid searches."""
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
import runpy

from src.config import ExperimentConfig


# Deliberately exclude latent_dim, mixture priors, likelihood variance and penalties.
ALLOWED_GRID_KEYS = {"hidden_dims", "activation", "learning_rate", "batch_size"}


@dataclass
class GridSearchConfig:
    base: ExperimentConfig = field(default_factory=ExperimentConfig)
    grid: dict = field(default_factory=lambda: {
        "hidden_dims": [(64,), (128, 128), (128, 128, 128)],
        "activation": ["relu", "tanh"],
        "learning_rate": [3e-4, 1e-3],
        "batch_size": [256, 1024],
    })
    screening_epochs: int = 30
    screening_patience: int = 10
    finalists: int = 2
    confirmation_epochs: int = 100
    confirmation_seeds: tuple[int, ...] = (42, 43, 44)
    workers: int = 2
    threads_per_worker: int = 1
    output_dir: str = "outputs/neural_grid"
    baseline_dir: str = "outputs/vowels/20260924_164535_c90671"


def load_search_config(path):
    config = runpy.run_path(str(Path(path).resolve()))["SEARCH"]
    if not isinstance(config, GridSearchConfig):
        raise TypeError("Define SEARCH = GridSearchConfig(...).")
    return config


def grid_candidates(config):
    if set(config.grid) != ALLOWED_GRID_KEYS:
        raise ValueError(f"Grid keys must be exactly {sorted(ALLOWED_GRID_KEYS)}; statistical changes are forbidden.")
    if any(not values for values in config.grid.values()):
        raise ValueError("Every grid axis needs at least one value.")
    if min(config.screening_epochs, config.confirmation_epochs, config.workers,
           config.threads_per_worker, config.finalists) < 1:
        raise ValueError("Search budgets and counts must be positive.")
    if not config.confirmation_seeds or len(set(config.confirmation_seeds)) != len(config.confirmation_seeds):
        raise ValueError("Confirmation seeds must be distinct and nonempty.")
    baseline = {"hidden_dims": config.base.model.hidden_dims, "activation": "relu",
                "learning_rate": config.base.training.learning_rate,
                "batch_size": config.base.training.batch_size}
    candidates = [baseline]
    for values in product(*config.grid.values()):
        candidate = dict(zip(config.grid, values))
        candidate["hidden_dims"] = tuple(candidate["hidden_dims"])
        if candidate not in candidates:
            candidates.append(candidate)
    for c in candidates:
        if c["activation"] not in ("relu", "tanh", "silu") or not c["hidden_dims"]:
            raise ValueError("Specify a supported activation and nonempty hidden_dims.")
        if min(c["hidden_dims"]) < 1 or c["learning_rate"] <= 0 or c["batch_size"] < 1:
            raise ValueError("Widths, learning rates and batch sizes must be positive.")
    return [{"candidate_id": f"c{i:03d}", **c} for i, c in enumerate(candidates)]
