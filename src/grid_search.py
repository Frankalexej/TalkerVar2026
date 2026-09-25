"""Resumable neural-only screening, multi-seed confirmation, then held-out scoring.

Workers receive train/validation arrays only. Test evaluation occurs after the
winning configuration has been committed to selection.json. No test leaderboard
is constructed for discarded candidates.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from pprint import pformat
import sys
import time
from types import SimpleNamespace
import uuid

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from src.config import DataConfig, ExperimentConfig, ModelConfig, NeuralModelConfig, TrainConfig, project_path
from src.data import prepare_data
from src.evaluation import fit_alignment, predict_neural, score_predictions
from src.experiment import write_json
from src.models import NeuralVaDE
from src.search_config import grid_candidates
from src.training import fit_neural, seed_everything


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def selection_score(scores):
    """Equal importance on a multiplicative scale; validation labels only."""
    return float(np.sqrt(scores["phoneme"]["accuracy"] * scores["speaker"]["accuracy"]))


def build_trial_config(search, candidate, seed, stage):
    config = deepcopy(search.base)
    config.model = NeuralModelConfig(**asdict(search.base.model), activation=candidate["activation"])
    config.model.hidden_dims = tuple(candidate["hidden_dims"])
    config.training.learning_rate = candidate["learning_rate"]
    config.training.batch_size = candidate["batch_size"]
    config.training.num_threads = search.threads_per_worker
    config.training.seed = seed
    config.training.epochs = search.screening_epochs if stage == "screen" else search.confirmation_epochs
    if stage == "screen":
        config.training.patience = search.screening_patience
    config.methods = ("gmvae_neural",)
    return config


def trial_worker(config_dict, cache_path, directory):
    """Pickle-safe worker entry point, importable from the notebook or Windows CLI."""
    directory = Path(directory)
    metrics_path = directory / "validation.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", config_dict)
    config = ExperimentConfig(data=DataConfig(**config_dict["data"]),
                              model=NeuralModelConfig(**config_dict["model"]),
                              training=TrainConfig(**config_dict["training"]),
                              methods=("gmvae_neural",))
    started = time.perf_counter()
    data = SimpleNamespace(**joblib.load(cache_path))
    assert set(data.x) == set(data.y) == {"train", "val"}, "Search worker must not receive test data."
    with (directory / "training.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=config.training.num_threads):
            seed_everything(config.training.seed, config.training.num_threads)
            model = NeuralVaDE(data.x["train"].shape[1], len(data.phonemes), len(data.speakers), config.model)
            model, training = fit_neural(model, "gmvae", data, config, directory)
            raw_train, _ = predict_neural(model, "gmvae", data.x["train"], config.training.batch_size, config.training.seed + 200)
            raw_val, _ = predict_neural(model, "gmvae", data.x["val"], config.training.batch_size, config.training.seed + 200)
            mappings = [fit_alignment(data.y["train"][:, i], raw_train[:, i], k)
                        for i, k in enumerate((len(data.phonemes), len(data.speakers)))]
            predicted = np.column_stack([mappings[i][raw_val[:, i]] for i in range(2)])
            scores = score_predictions(data.y["val"], predicted, (len(data.phonemes), len(data.speakers)))
            torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "model_config": asdict(config.model), "model_class": "NeuralVaDE",
                        "input_dim": data.x["train"].shape[1], "phonemes": data.phonemes,
                        "speakers": data.speakers, "label_mappings": [m.tolist() for m in mappings],
                        "training": training}, directory / "best.pt")
    result = {"validation_score": selection_score(scores), "val_phoneme_accuracy": scores["phoneme"]["accuracy"],
              "val_speaker_accuracy": scores["speaker"]["accuracy"], "val_joint_accuracy": scores["joint_accuracy"],
              "validation": scores, "training": training, "elapsed_seconds": time.perf_counter() - started,
              "n_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(metrics_path, result)  # Completion marker written last.
    return result


def execute_stage(search, candidates, seeds, stage, run_dir):
    rows = []
    jobs = []
    for candidate in candidates:
        for seed in seeds:
            config = build_trial_config(search, candidate, seed, stage)
            directory = run_dir / stage / f"{candidate['candidate_id']}_seed_{seed}"
            jobs.append((candidate, seed, config, directory))
    print(f"{stage}: {len(jobs)} runs; {search.workers} workers, {search.threads_per_worker} thread(s) each", flush=True)
    with ProcessPoolExecutor(max_workers=search.workers) as pool:
        pending = {pool.submit(trial_worker, asdict(cfg), str(run_dir / "train_val.joblib"), str(directory)):
                   (candidate, seed, directory) for candidate, seed, cfg, directory in jobs}
        for future in as_completed(pending):
            candidate, seed, directory = pending[future]
            result = future.result()
            rows.append({**candidate, "seed": seed, "stage": stage, "trial_dir": str(directory.relative_to(run_dir)),
                         **{k: v for k, v in result.items() if k not in ("validation", "training")},
                         "best_epoch": result["training"]["best_epoch"]})
            pd.DataFrame(rows).sort_values(["validation_score", "candidate_id"], ascending=[False, True]).to_csv(
                run_dir / f"{stage}_leaderboard.csv", index=False)
            print(f"  {len(rows)}/{len(jobs)} {candidate['candidate_id']} seed={seed}: "
                  f"val phoneme={result['val_phoneme_accuracy']:.1%}, speaker={result['val_speaker_accuracy']:.1%}, "
                  f"score={result['validation_score']:.4f}, {result['elapsed_seconds']:.1f}s", flush=True)
    return pd.DataFrame(rows)


def load_neural_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = NeuralModelConfig(**checkpoint["model_config"])
    model = NeuralVaDE(checkpoint["input_dim"], len(checkpoint["phonemes"]), len(checkpoint["speakers"]), config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_selected(search, run_dir, data, winner_id):
    """Called only after selection.json exists; score control and winner, all seeds."""
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["winner_id"] == winner_id
    rows = []
    for candidate_id in dict.fromkeys(("c000", winner_id)):
        for seed in search.confirmation_seeds:
            directory = run_dir / "confirm" / f"{candidate_id}_seed_{seed}"
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            model, checkpoint = load_neural_checkpoint(directory / "best.pt")
            raw, _ = predict_neural(model, "gmvae", data.x["test"], config["training"]["batch_size"], seed + 200)
            mappings = [np.array(m) for m in checkpoint["label_mappings"]]
            prediction = np.column_stack([mappings[c][raw[:, c]] for c in range(2)])
            scores = score_predictions(data.y["test"], prediction, (len(data.phonemes), len(data.speakers)))
            pd.DataFrame({"source_row": data.frame.iloc[data.indices["test"]].source_row.to_numpy(),
                          "true_phoneme": data.y["test"][:, 0], "true_speaker": data.y["test"][:, 1],
                          "pred_phoneme": prediction[:, 0], "pred_speaker": prediction[:, 1]}).to_csv(
                              directory / "test_predictions.csv", index=False)
            write_json(directory / "test_metrics.json", scores)
            rows.append({"candidate_id": candidate_id, "seed": seed,
                         "phoneme_accuracy": scores["phoneme"]["accuracy"],
                         "speaker_accuracy": scores["speaker"]["accuracy"], "joint_accuracy": scores["joint_accuracy"]})
    result = pd.DataFrame(rows)
    result.to_csv(run_dir / "test_results.csv", index=False)
    means = result.groupby("candidate_id")[["phoneme_accuracy", "speaker_accuracy", "joint_accuracy"]].agg(["mean", "std"])
    means.to_csv(run_dir / "test_summary.csv")
    pairs = result[result.candidate_id == winner_id].merge(result[result.candidate_id == "c000"], on="seed", suffixes=("_winner", "_control"))
    deltas = {name: (pairs[f"{name}_winner"] - pairs[f"{name}_control"]).tolist()
              for name in ("phoneme_accuracy", "speaker_accuracy", "joint_accuracy")}
    write_json(run_dir / "paired_seed_differences.json", deltas)
    print("Final held-out results (configuration chosen using validation only):\n" + result.to_string(index=False), flush=True)
    return result


def export_config(search, winner, seed, run_dir):
    config = build_trial_config(search, winner, seed, "confirm")
    config.output_dir = "outputs/neural_grid_winner"
    body = ("# Validation-selected configuration. Statistical model is unchanged.\n"
            "from src.config import DataConfig, ExperimentConfig, NeuralModelConfig, TrainConfig\n\n"
            "CONFIG = ExperimentConfig(\n"
            f"    data=DataConfig(**{pformat(asdict(config.data))}),\n"
            f"    model=NeuralModelConfig(**{pformat(asdict(config.model))}),\n"
            f"    training=TrainConfig(**{pformat(asdict(config.training))}),\n"
            "    methods=('gmvae_neural',),\n"
            f"    output_dir={config.output_dir!r},\n)\n")
    (run_dir / "best_config.py").write_text(body, encoding="utf-8")


def run_grid(search, resume=None):
    if type(search.base.model) is not ModelConfig:
        raise TypeError("The grid must start from the original ModelConfig, not the shared-w model.")
    candidates = grid_candidates(search)
    data = prepare_data(search.base.data)
    baseline = project_path(search.baseline_dir)
    if (baseline / "config.json").exists():
        old = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
        fixed = asdict(search.base.model)
        fixed.pop("hidden_dims")
        assert all(old["model"][k] == v for k, v in fixed.items()), "Statistical settings must match the baseline."
        old_data = json.loads((baseline / "data.json").read_text(encoding="utf-8"))
        for key in ("csv_sha256", "speakers", "phonemes", "features", "log_features", "imputation_medians", "scaler_mean", "scaler_scale"):
            assert data.metadata[key] == old_data[key], f"Baseline data mismatch: {key}"
    manifest = data.frame[["source_row", "speaker_id", "label"]].copy()
    for part, indices in data.indices.items():
        manifest.loc[indices, "split"] = part
    if (baseline / "split_manifest.csv").exists():
        pd.testing.assert_frame_equal(manifest, pd.read_csv(baseline / "split_manifest.csv"))
    plan = {"search": asdict(search), "csv_sha256": data.metadata["csv_sha256"], "candidates": candidates,
            "selection_metric": "geometric mean of validation phoneme and speaker accuracy",
            "source_hashes": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ("grid_search.py", "models.py", "training.py", "data.py", "evaluation.py")}}
    if resume is None:
        run_dir = project_path(search.output_dir) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        run_dir.mkdir(parents=True)
        write_json(run_dir / "plan.json", plan)
        write_json(run_dir / "data.json", data.metadata)
        manifest.to_csv(run_dir / "split_manifest.csv", index=False)
        joblib.dump({"x": {p: data.x[p] for p in ("train", "val")},
                     "y": {p: data.y[p] for p in ("train", "val")},
                     "phonemes": data.phonemes, "speakers": data.speakers}, run_dir / "train_val.joblib")
        joblib.dump({"imputer": data.imputer, "scaler": data.scaler, "features": search.base.data.features,
                     "log_features": search.base.data.log_features}, run_dir / "preprocessing.joblib")
        write_json(run_dir / "environment.json", {"executable": sys.executable, "torch": torch.__version__,
                                                   "python": sys.version})
    else:
        run_dir = project_path(str(resume))
        old_plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        if json_digest(old_plan) != json_digest(plan):
            raise ValueError("Cannot resume: configuration, dataset or source code changed.")
        if (run_dir / "complete.json").exists():
            print(f"Search already complete: {run_dir}", flush=True)
            return run_dir
    print(f"GRID DIRECTORY: {run_dir}", flush=True)
    screen = execute_stage(search, candidates, [search.base.training.seed], "screen", run_dir)
    ranking = screen.sort_values(["validation_score", "candidate_id"], ascending=[False, True])
    finalists = ranking.loc[ranking.candidate_id != "c000", "candidate_id"].head(search.finalists).tolist()
    confirmed = [c for c in candidates if c["candidate_id"] in ("c000", *finalists)]
    write_json(run_dir / "shortlist.json", {"candidate_ids": [c["candidate_id"] for c in confirmed],
                                            "based_on": "screening validation only"})
    confirmation = execute_stage(search, confirmed, search.confirmation_seeds, "confirm", run_dir)
    final_ranking = confirmation.groupby("candidate_id").agg(
        mean_validation_score=("validation_score", "mean"), std_validation_score=("validation_score", "std"),
        mean_val_phoneme_accuracy=("val_phoneme_accuracy", "mean"), mean_val_speaker_accuracy=("val_speaker_accuracy", "mean"))
    final_ranking = final_ranking.sort_values(["mean_validation_score", "candidate_id"], ascending=[False, True])
    final_ranking.to_csv(run_dir / "confirmation_ranking.csv")
    winner_id = final_ranking.index[0]
    winner = next(c for c in candidates if c["candidate_id"] == winner_id)
    selection = {"winner_id": winner_id, "winner": winner,
                 "selection_basis": "mean validation geometric accuracy across confirmation seeds",
                 "test_accessed_during_selection": False}
    write_json(run_dir / "selection.json", selection)
    export_config(search, winner, search.confirmation_seeds[0], run_dir)
    seed_everything(search.base.training.seed, search.threads_per_worker)
    with threadpool_limits(limits=search.threads_per_worker):
        evaluate_selected(search, run_dir, data, winner_id)
    write_json(run_dir / "complete.json", {"winner_id": winner_id, "screening_trials": len(screen),
                                            "confirmation_trials": len(confirmation)})
    return run_dir
