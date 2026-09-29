"""Prespecified design for the dual-codebook experiment."""
from dataclasses import dataclass, field
from src.config import DataConfig


@dataclass
class VQConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training_speakers: tuple[str, ...] = ('jvs003', 'jvs009', 'jvs018', 'jvs036', 'jvs037',
                                        'jvs038', 'jvs046', 'jvs061', 'jvs065', 'jvs078')
    heldout_count: int = 3
    hidden_dims: tuple[int, ...] = (128, 128)
    embedding_dim: int = 8
    commitment: float = 0.25
    lambdas: tuple[float, ...] = (0.1, 1.0, 10.0)
    selection_seeds: tuple[int, ...] = (42, 43, 44)
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    baseline_seeds: tuple[int, ...] = (42, 43, 44)
    selection_epochs: int = 40
    epochs: int = 80
    pretrain_epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 0.001
    workers: int = 2
    threads: int = 1
    output_dir: str = 'outputs/dual_vq'

