"""Main experiment: edit these settings, then rerun the notebook or CLI."""
from src.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig

CONFIG = ExperimentConfig(
    data=DataConfig(
        csv_path="metadata_vowels_acoustics.csv",
        features=("f0_median_hz", "f1_hz", "f2_hz", "f3_hz", "duration_s"),
        n_speakers=10,
        seed=2026,
        split=(0.8, 0.1, 0.1),
        log_features=False,
    ),
    model=ModelConfig(
        hidden_dims=(128, 128),
        latent_dim=8,
        interaction_weight=10.0,
        observation_variance=0.1,
        train_mc_samples=4,
        eval_mc_samples=32,
    ),
    training=TrainConfig(
        seed=42,
        device="cpu",
        num_threads=4,
        batch_size=512,
        epochs=100,
        pretrain_epochs=20,
        learning_rate=1e-3,
        patience=20,
    ),
    methods=("gmvae", "knn", "classifier"),
    knn_neighbors=15,
    output_dir="outputs/vowels",
)
