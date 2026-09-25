"""Validation-selected c022 from the 20260924_225129_ccc857 search.

Run: python src/train.py --config src/configs/neural_grid_winner.py
Seed 42 is the first prespecified confirmation seed, not a best-seed choice.
"""
from src.config import DataConfig, ExperimentConfig, NeuralModelConfig, TrainConfig

CONFIG = ExperimentConfig(
    data=DataConfig(
        csv_path="metadata_vowels_acoustics.csv",
        features=("f0_median_hz", "f1_hz", "f2_hz", "f3_hz", "duration_s"),
        n_speakers=10, seed=2026, split=(0.8, 0.1, 0.1), log_features=False,
    ),
    model=NeuralModelConfig(
        hidden_dims=(128, 128, 128), activation="tanh", latent_dim=8,
        interaction_weight=10.0, observation_variance=0.1,
        train_mc_samples=4, eval_mc_samples=32, learn_mixture_weights=False,
    ),
    training=TrainConfig(
        learning_rate=3e-4, batch_size=1024, epochs=100, pretrain_epochs=20,
        patience=20, min_delta=1e-4, grad_clip=10.0, weight_decay=0.0,
        seed=42, device="cpu", num_threads=1,
    ),
    methods=("gmvae_neural",),
    output_dir="outputs/neural_grid_winner",
)
