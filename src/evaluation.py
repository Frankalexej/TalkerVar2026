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
        if method in ("gmvae", "gmvae_neural", "gmvae_shared_w"):
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


@torch.no_grad()
def shared_w_diagnostics(model, x, batch_size):
    """Does the new posterior/prior use w? No semantic interpretation is assumed."""
    device = next(model.parameters()).device
    means, logvars = [], []
    model.eval()
    for start in range(0, len(x), batch_size):
        _, _, mean, logvar = model.posterior_parameters(torch.from_numpy(x[start:start + batch_size]).to(device))
        means.append(mean.cpu().numpy())
        logvars.append(logvar.cpu().numpy())
    mean, logvar = np.concatenate(means), np.concatenate(logvars)
    kl_per_dim = 0.5 * (mean ** 2 + np.exp(logvar) - 1 - logvar)
    grid = torch.cat([torch.zeros(1, model.config.w_dim), torch.eye(model.config.w_dim),
                      -torch.eye(model.config.w_dim)]).to(device)
    prior_mean, prior_logvar, _ = model.conditional_parameters(grid)
    return {
        "mean_kl_to_standard_normal_nats": float(kl_per_dim.sum(1).mean()),
        "kl_per_dimension_nats": kl_per_dim.mean(0).tolist(),
        "posterior_mean_std_across_tokens": mean.std(0).tolist(),
        "mean_posterior_std": np.exp(0.5 * logvar).mean(0).tolist(),
        "prior_mean_change_rms_at_plus_minus_unit_w": float((prior_mean[1:] - prior_mean[:1]).square().mean().sqrt()),
        "prior_logvar_change_rms_at_plus_minus_unit_w": float((prior_logvar[1:] - prior_logvar[:1]).square().mean().sqrt()),
    }
