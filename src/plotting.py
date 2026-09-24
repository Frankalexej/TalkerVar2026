"""Saved learning curves, accuracy comparison, and held-out confusion matrices."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_plots(run_dir):
    run_dir = Path(run_dir)
    summary = pd.read_csv(run_dir / "summary.csv")
    test = summary.loc[summary.split == "test"].set_index("method")
    ax = test[["phoneme_accuracy", "speaker_accuracy"]].plot.bar(figsize=(7, 4), rot=0)
    ax.set(ylim=(0, 1), ylabel="Test accuracy", title="Vowel category recognition")
    ax.legend(["Phoneme", "Speaker"])
    ax.figure.tight_layout()
    ax.figure.savefig(run_dir / "accuracy.png", dpi=160)
    plt.close(ax.figure)
    metadata = json.loads((run_dir / "data.json").read_text(encoding="utf-8"))
    for method in test.index:
        directory = run_dir / method
        history = directory / "history.csv"
        if history.exists():
            df = pd.read_csv(history)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(df.epoch, df.train_loss, label="Train")
            ax.plot(df.epoch, df.val_loss, label="Validation")
            ax.set(xlabel="Epoch", ylabel="Objective", title=method)
            ax.legend()
            fig.tight_layout()
            fig.savefig(directory / "learning_curve.png", dpi=160)
            plt.close(fig)
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, name, labels in zip(axes, ("phoneme", "speaker"), (metadata["phonemes"], metadata["speakers"])):
            cm = np.array(metrics["test"][name]["confusion_matrix"])
            normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
                   yticklabels=labels, xlabel="Predicted", ylabel="True", title=f"{method}: {name}")
            ax.tick_params(axis="x", rotation=60)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(directory / "confusion_matrices.png", dpi=160)
        plt.close(fig)
