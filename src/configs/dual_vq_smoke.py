from src.vq_config import VQConfig

CONFIG = VQConfig(lambdas=(0.1,), selection_seeds=(42,), seeds=(42,), baseline_seeds=(),
                  selection_epochs=1, epochs=1, pretrain_epochs=1, output_dir='outputs/dual_vq_smoke')
