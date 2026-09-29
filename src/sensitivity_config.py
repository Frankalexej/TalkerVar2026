"""Fixed design: codebook capacity, quantized speaker CE and raw clusterability."""
from dataclasses import dataclass
from src.vq_config import VQConfig


@dataclass
class SensitivityConfig(VQConfig):
    codebook_pairs: tuple[tuple[int, int], ...] = ((5, 10), (64, 64))
    lambdas: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)
    kmeans_n_init: int = 10
    gmm_n_init: int = 5
    gmm_max_iter: int = 300
    gmm_reg_covar: float = 1e-5
    reference_run: str = 'outputs/dual_vq/20260928_190154_268b85'
    output_dir: str = 'outputs/vq_sensitivity'

