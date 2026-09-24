"""One entry point shared by the command line and notebook."""
from dataclasses import asdict
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import uuid

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier

from src.config import ExperimentConfig, project_path
from src.data import prepare_data
from src.evaluation import fit_alignment, mixture_diagnostics, predict_neural, score_predictions
from src.models import TwoFactorVaDE, VowelClassifier
from src.training import fit_neural, seed_everything


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiment(config: ExperimentConfig, data=None) -> Path:
    """Train selected methods and save a fresh, self-contained run directory."""
    if set(config.methods) - {"gmvae", "knn", "classifier"} or not config.methods:
        raise ValueError("methods must select gmvae, knn and/or classifier.")
    if config.training.epochs < 1 or config.training.pretrain_epochs < 0:
        raise ValueError("epochs must be positive and pretrain_epochs nonnegative.")
    if config.model.observation_variance <= 0 or min(config.model.train_mc_samples, config.model.eval_mc_samples) < 1:
        raise ValueError("Observation variance and Monte Carlo sample counts must be positive.")
    seed_everything(config.training.seed, config.training.num_threads)
    data = prepare_data(config.data) if data is None else data
    # Timestamp plus random suffix avoids overwriting previous experiments.
    run_dir = project_path(config.output_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", asdict(config))
    write_json(run_dir / "data.json", data.metadata)
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib")}
    write_json(run_dir / "environment.json", {"executable": sys.executable, "python": sys.version,
               "platform": platform.platform(), "versions": versions})
    manifest = data.frame[["source_row", "speaker_id", "label"]].copy()
    for part, ix in data.indices.items():
        manifest.loc[ix, "split"] = part
    manifest.to_csv(run_dir / "split_manifest.csv", index=False)
    joblib.dump({"imputer": data.imputer, "scaler": data.scaler,
                 "features": config.data.features, "log_features": config.data.log_features,
                 "phonemes": data.phonemes, "speakers": data.speakers}, run_dir / "preprocessing.joblib")
    print(f"Run: {run_dir}\nSpeakers: {', '.join(data.speakers)}\nRows: {data.metadata['split_counts']}", flush=True)
    counts = (len(data.phonemes), len(data.speakers))
    rows, report = [], {}
    for method in config.methods:
        started = time.perf_counter()
        seed_everything(config.training.seed, config.training.num_threads)
        method_dir = run_dir / method
        method_dir.mkdir()
        print(f"Training {method}...", flush=True)
        if method == "knn":
            if config.knn_neighbors > len(data.x["train"]) or config.knn_neighbors < 1:
                raise ValueError("knn_neighbors must be between 1 and the training set size.")
            model = KNeighborsClassifier(n_neighbors=config.knn_neighbors, weights="uniform", n_jobs=config.training.num_threads)
            model.fit(data.x["train"], data.y["train"])
            raw = {part: model.predict(data.x[part]) for part in ("val", "test")}
            probabilities = {}
            details = {"n_neighbors": config.knn_neighbors, "supervised": True}
            joblib.dump(model, method_dir / "model.joblib")
        else:
            cls = TwoFactorVaDE if method == "gmvae" else VowelClassifier
            model = cls(len(config.data.features), *counts, config.model)
            model, details = fit_neural(model, method, data, config, method_dir)
            parts = ("train", "val", "test") if method == "gmvae" else ("val", "test")
            predictions = {part: predict_neural(model, method, data.x[part], config.training.batch_size,
                                               config.training.seed + 200) for part in parts}
            raw = {part: value[0] for part, value in predictions.items()}
            probabilities = {part: value[1] for part, value in predictions.items()}
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "method": method, "model_config": asdict(config.model), "input_dim": len(config.data.features),
                        "phonemes": data.phonemes, "speakers": data.speakers,
                        "training": details}, method_dir / "best.pt")
        mappings = None
        if method == "gmvae":
            mappings = [fit_alignment(data.y["train"][:, c], raw["train"][:, c], size)
                        for c, size in enumerate(counts)]
            details["train_label_mappings"] = [m.tolist() for m in mappings]
            write_json(method_dir / "label_mappings.json", details["train_label_mappings"])
        for part in ("val", "test"):
            pred = raw[part] if mappings is None else np.column_stack([
                mappings[c][raw[part][:, c]] for c in range(2)])
            scores = score_predictions(data.y[part], pred, counts)
            if method == "gmvae":
                scores["mixture_diagnostics"] = mixture_diagnostics(probabilities[part])
                # Descriptive clustering statistic only; never used for selection.
                oracle = np.column_stack([fit_alignment(data.y[part][:, c], raw[part][:, c], size)[raw[part][:, c]]
                                          for c, size in enumerate(counts)])
                scores["split_optimal_alignment_accuracy"] = {
                    name: float((oracle[:, c] == data.y[part][:, c]).mean())
                    for c, name in enumerate(("phoneme", "speaker"))}
            details[part] = scores
            saved = pd.DataFrame({"source_row": data.frame.iloc[data.indices[part]].source_row.to_numpy(),
                                  "true_phoneme": data.y[part][:, 0], "true_speaker": data.y[part][:, 1],
                                  "pred_phoneme": pred[:, 0], "pred_speaker": pred[:, 1],
                                  "raw_category_1": raw[part][:, 0], "raw_category_2": raw[part][:, 1]})
            saved.to_csv(method_dir / f"{part}_predictions.csv", index=False)
            rows.append({"method": method, "split": part,
                         "phoneme_accuracy": scores["phoneme"]["accuracy"],
                         "speaker_accuracy": scores["speaker"]["accuracy"],
                         "joint_accuracy": scores["joint_accuracy"],
                         "phoneme_balanced_accuracy": scores["phoneme"]["balanced_accuracy"],
                         "speaker_balanced_accuracy": scores["speaker"]["balanced_accuracy"]})
        details["elapsed_seconds"] = time.perf_counter() - started
        report[method] = details
        write_json(method_dir / "metrics.json", details)
        print(f"  test: phoneme={details['test']['phoneme']['accuracy']:.3%}, "
              f"speaker={details['test']['speaker']['accuracy']:.3%}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    write_json(run_dir / "metrics.json", report)
    from src.plotting import save_plots
    save_plots(run_dir)
    print(summary.to_string(index=False), flush=True)
    return run_dir
