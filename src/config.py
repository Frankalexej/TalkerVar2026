"""Small typed configs, loaded from ordinary Python files."""
from dataclasses import dataclass, field
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataConfig:
    csv_path: str = "metadata_vowels_acoustics.csv"
    features: tuple[str, ...] = ("f0_median_hz", "f1_hz", "f2_hz", "f3_hz", "duration_s")
    n_speakers: int = 10
    speaker_ids: tuple[str, ...] | None = None  # Set to reuse an explicit selection.
    seed: int = 2026
    split: tuple[float, float, float] = (0.8, 0.1, 0.1)
    log_features: bool = False


@dataclass
class ModelConfig:
    hidden_dims: tuple[int, ...] = (128, 128)
    latent_dim: int = 8
    interaction_weight: float = 10.0
    observation_variance: float = 0.1
    train_mc_samples: int = 4
    eval_mc_samples: int = 32
    learn_mixture_weights: bool = False


@dataclass
class NeuralModelConfig(ModelConfig):
    """Only the hidden-layer activation changes beyond the existing width setting."""
    activation: str = "relu"


@dataclass
class SharedWModelConfig(ModelConfig):
    """Additional settings for the shared-w extension; old defaults are untouched."""
    w_dim: int = 2  # One vector-valued w per token, shared by BOTH category factors.
    prior_hidden_dims: tuple[int, ...] = (64,)
    objective: str = "paper"  # "paper" (Eq. 5 estimator) or "structured" (exact ELBO MC).


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cpu"  # "auto", "cpu", or "cuda"
    num_threads: int = 4
    batch_size: int = 512
    epochs: int = 100
    pretrain_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 20
    min_delta: float = 1e-4
    grad_clip: float = 10.0


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    methods: tuple[str, ...] = ("gmvae", "knn", "classifier")
    knn_neighbors: int = 15
    output_dir: str = "outputs/vowels"


def load_config(path: str | Path) -> ExperimentConfig:
    """Execute a trusted Python config defining CONFIG."""
    config = runpy.run_path(str(Path(path).resolve()))["CONFIG"]
    if not isinstance(config, ExperimentConfig):
        raise TypeError("The config file must define CONFIG = ExperimentConfig(...).")
    return config


def project_path(path: str) -> Path:
    result = Path(path).expanduser()
    return result if result.is_absolute() else ROOT / result
