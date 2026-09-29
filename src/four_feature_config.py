"""Four-feature controlled experiment; previous objectives are not retuned."""
from dataclasses import dataclass, field
from src.config import DataConfig, ModelConfig, TrainConfig
from src.sensitivity_config import SensitivityConfig


@dataclass
class FourFeatureConfig(SensitivityConfig):
    data: DataConfig = field(default_factory=lambda: DataConfig(features=('f1_hz','f2_hz','f3_hz','duration_s')))
    reference_run: str = 'outputs/vq_sensitivity/20260929_011130_ec2936'
    codebook_pairs: tuple = ((16,16),(64,64))
    lambdas: tuple = (0.0,1.0)
    gmvae_model: ModelConfig = field(default_factory=ModelConfig)
    baseline_training: TrainConfig = field(default_factory=lambda: TrainConfig(num_threads=1))
    gmvae_device: str = 'cuda'  # Execution device only; original optimizer/objective are unchanged.
    output_dir: str = 'outputs/four_feature'
