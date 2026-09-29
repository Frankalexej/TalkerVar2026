from src.sensitivity_config import SensitivityConfig

CONFIG = SensitivityConfig(seeds=(42,), epochs=1, pretrain_epochs=1,
                          lambdas=(0.0, 0.1), kmeans_n_init=1, gmm_n_init=1,
                          gmm_max_iter=30, output_dir='outputs/vq_sensitivity_smoke')

