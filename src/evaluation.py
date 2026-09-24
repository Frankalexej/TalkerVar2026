"""Shared metrics and train-calibrated category alignment."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             adjusted_rand_score, normalized_mutual_info_score)


def fit_alignment(truth: np.ndarray, clusters: np.ndarray, n_categories: int) -> np.ndarray:
    """One-to-one Hungarian mapping fitted on training labels only."""
    counts = np.zeros((n_categories, n_categories), dtype=np.int64)
    np.add.at(counts, (clusters, truth), 1)
    rows, columns = linear_sum_assignment(-counts)
    mapping = np.empty(n_categories, dtype=np.int64)
    mapping[rows] = columns
    return mapping


def score_predictions(truth, predictions, category_counts):
    result = {"joint_accuracy": float(np.all(truth == predictions, axis=1).mean())}
    for column, (name, size) in enumerate(zip(("phoneme", "speaker"), category_counts)):
        y, pred = truth[:, column], predictions[:, column]
        result[name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "ari": float(adjusted_rand_score(y, pred)),
            "nmi": float(normalized_mutual_info_score(y, pred)),
            "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(size)).tolist(),
        }
    return result


@torch.no_grad()
def predict_neural(model, method, x, batch_size, seed):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    predictions, probabilities = [], []
    model.eval()
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start:start + batch_size]).to(device)
        if method == "gmvae":
            q = model.category_probabilities(batch, generator=generator)
            predictions.append(torch.stack([q.sum(2).argmax(1), q.sum(1).argmax(1)], 1).cpu().numpy())
            probabilities.append(q.cpu().numpy())
        else:
            p, s = model(batch)
            predictions.append(torch.stack([p.argmax(1), s.argmax(1)], 1).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(probabilities) if probabilities else None


def mixture_diagnostics(q):
    def describe(probabilities):
        mean = probabilities.mean(0)
        entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(1).mean()
        marginal_entropy = -(mean * np.log(np.maximum(mean, 1e-12))).sum()
        return {"mean_probabilities": mean.tolist(), "conditional_entropy_nats": float(entropy),
                "mutual_information_nats": float(marginal_entropy - entropy),
                "active_argmax_categories": int(len(np.unique(probabilities.argmax(1))))}
    return {"phoneme": describe(q.sum(2)), "speaker": describe(q.sum(1)),
            "joint": describe(q.reshape(len(q), -1))}
