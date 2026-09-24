"""Speaker selection, disjoint row splits, and training-only preprocessing."""
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DataConfig, project_path


@dataclass
class VowelData:
    frame: pd.DataFrame
    indices: dict[str, np.ndarray]
    x: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    phonemes: list[str]
    speakers: list[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    metadata: dict


def prepare_data(config: DataConfig) -> VowelData:
    path = project_path(config.csv_path)
    frame = pd.read_csv(path, dtype={"speaker_id": str, "label": str})
    required = [*config.features, "speaker_id", "label"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    if frame[["speaker_id", "label"]].isna().any().any():
        raise ValueError("Speaker and phoneme labels cannot be missing.")
    available = sorted(frame.speaker_id.unique())
    if config.speaker_ids is None:
        if not 1 <= config.n_speakers <= len(available):
            raise ValueError("n_speakers must be between 1 and the available speaker count.")
        speakers = sorted(np.random.default_rng(config.seed).choice(
            available, config.n_speakers, replace=False).tolist())
    else:
        speakers = sorted(config.speaker_ids)
        if not speakers or len(set(speakers)) != len(speakers) or not set(speakers) <= set(available):
            raise ValueError("speaker_ids must contain distinct existing speakers.")
    frame = frame.loc[frame.speaker_id.isin(speakers)].copy()
    frame.insert(0, "source_row", frame.index)  # Zero-based CSV data row, excluding header.
    frame.reset_index(drop=True, inplace=True)
    phonemes = sorted(frame.label.unique())
    labels = np.column_stack([
        pd.Categorical(frame.label, categories=phonemes).codes,
        pd.Categorical(frame.speaker_id, categories=speakers).codes,
    ]).astype(np.int64)
    fractions = np.asarray(config.split, dtype=float)
    if len(fractions) != 3 or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("split must contain three positive fractions summing to one.")
    # Joint stratification keeps both speaker and phoneme proportions comparable.
    strata = frame.speaker_id + ":" + frame.label
    all_indices = np.arange(len(frame))
    train, holdout = train_test_split(
        all_indices, test_size=float(fractions[1:].sum()),
        random_state=config.seed, stratify=strata,
    )
    val, test = train_test_split(
        holdout, test_size=float(fractions[2] / fractions[1:].sum()),
        random_state=config.seed + 1, stratify=strata.iloc[holdout],
    )
    indices = {"train": train, "val": val, "test": test}
    raw = frame[list(config.features)].apply(pd.to_numeric, errors="raise").to_numpy(float)
    # Nonpositive acoustic measures are invalid. Retain their rows and impute them.
    raw[~np.isfinite(raw) | (raw <= 0)] = np.nan
    if config.log_features:
        raw = np.log(raw)
    if np.isnan(raw[train]).all(axis=0).any():
        raise ValueError("At least one feature has no observed values in training.")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaler.fit(imputer.fit_transform(raw[train]))
    x = {part: scaler.transform(imputer.transform(raw[ix])).astype(np.float32)
         for part, ix in indices.items()}
    metadata = {
        "csv_path": str(path), "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "features": list(config.features), "log_features": config.log_features,
        "speaker_seed": config.seed, "speakers": speakers, "phonemes": phonemes,
        "selected_rows": len(frame), "split_counts": {key: len(ix) for key, ix in indices.items()},
        "missing_or_invalid": dict(zip(config.features, np.isnan(raw).sum(axis=0).tolist())),
        "imputation_medians": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "split_unit": "vowel token; stratified by speaker x phoneme",
    }
    return VowelData(frame, indices, x, {p: labels[ix] for p, ix in indices.items()},
                     phonemes, speakers, imputer, scaler, metadata)
