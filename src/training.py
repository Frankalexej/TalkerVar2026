"""Neural training, unlabeled initialization, and validation early stopping."""
from copy import deepcopy
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

from src.config import ExperimentConfig
from src.data import VowelData
from src.models import TwoFactorVaDE


def seed_everything(seed: int, num_threads: int):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(num_threads))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment.")
    return torch.device(name)


def make_loader(data: VowelData, part: str, batch_size: int, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(data.x[part]), torch.from_numpy(data.y[part]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=(part == "train"),
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def initialize_gmvae(model: TwoFactorVaDE, data: VowelData, config: ExperimentConfig, device):
    """Reconstruct training x, then initialize additive means with residual KMeans.

    Initialization sees acoustic vectors only. First KMeans discovers K_phoneme
    centers, then K_speaker centers in their residuals. Axis ordering is an
    explicit initialization assumption, not supervision or a semantic guarantee.
    """
    loader = make_loader(data, "train", config.training.batch_size, config.training.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    records = []
    model.train()
    for epoch in range(1, config.training.pretrain_epochs + 1):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            mean, _ = model.encode(x)
            loss = (model.decoder(mean) - x).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            optimizer.step()
            total += loss.item() * len(x)
        records.append({"epoch": epoch, "train_mse": total / len(loader.dataset)})
        if epoch == 1 or epoch % 10 == 0:
            print(f"  autoencoder {epoch:3d}: MSE={records[-1]['train_mse']:.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        batches = np.array_split(data.x["train"], max(1, int(np.ceil(len(data.x["train"]) / 2048))))
        latent = torch.cat([model.encode(torch.from_numpy(batch).to(device))[0].cpu()
                            for batch in batches]).numpy()
    first = KMeans(n_clusters=model.n_phonemes, n_init=10, random_state=config.training.seed).fit(latent)
    residual = latent - first.cluster_centers_[first.labels_]
    second = KMeans(n_clusters=model.n_speakers, n_init=10, random_state=config.training.seed + 1).fit(residual)
    a, b = first.cluster_centers_, second.cluster_centers_
    prior_variance = np.maximum((residual - b[second.labels_]).var(0), 0.05)
    with torch.no_grad():
        model.global_mean.copy_(torch.as_tensor(a.mean(0) + b.mean(0), device=device))
        model.phoneme_effect.copy_(torch.as_tensor(a - a.mean(0), device=device))
        model.speaker_effect.copy_(torch.as_tensor(b - b.mean(0), device=device))
        model.interaction.zero_()
        model.component_logvar.copy_(torch.as_tensor(np.log(prior_variance), device=device).expand_as(model.component_logvar))
        model.z_logvar.weight.zero_()
        model.z_logvar.bias.copy_(torch.as_tensor(np.log(prior_variance * 0.1), device=device))
    return pd.DataFrame(records)


def fit_neural(model, method: str, data: VowelData, config: ExperimentConfig, output_dir):
    settings = config.training
    device = resolve_device(settings.device)
    model.to(device)
    if method in ("gmvae", "gmvae_neural", "gmvae_shared_w"):
        initialization = initialize_gmvae(model, data, config, device)
        initialization.to_csv(output_dir / "pretraining_history.csv", index=False)
    loaders = {part: make_loader(data, part, settings.batch_size, settings.seed)
               for part in ("train", "val")}
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate,
                                 weight_decay=settings.weight_decay)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state, records = None, []
    for epoch in range(1, settings.epochs + 1):
        record = {"epoch": epoch}
        for part, loader in loaders.items():
            training = part == "train"
            model.train(training)
            # Common random numbers make validation scores comparable across epochs.
            generator = None if training else torch.Generator(device=device).manual_seed(settings.seed + 100)
            totals = {}
            with torch.set_grad_enabled(training):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    loss, components = (model.loss(x, generator=generator) if method in ("gmvae", "gmvae_neural", "gmvae_shared_w")
                                        else model.loss(x, y))
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite {method} loss at epoch {epoch}.")
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
                        optimizer.step()
                    for key, value in components.items():
                        totals[key] = totals.get(key, 0.0) + value.item() * len(x)
            record.update({f"{part}_{key}": total / len(loader.dataset) for key, total in totals.items()})
        records.append(record)
        if record["val_loss"] < best_loss - settings.min_delta:
            best_loss, best_epoch, stale = record["val_loss"], epoch, 0
            best_state = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"  {method:10s} {epoch:3d}: train={record['train_loss']:.4f} val={record['val_loss']:.4f}", flush=True)
        if stale >= settings.patience:
            print(f"  early stopping at {epoch}; best epoch {best_epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint. Set epochs >= 1.")
    model.load_state_dict(best_state)
    model.eval()
    pd.DataFrame(records).to_csv(output_dir / "history.csv", index=False)
    return model, {"best_epoch": best_epoch, "best_validation_loss": best_loss,
                   "epochs_completed": len(records), "device": str(device)}
