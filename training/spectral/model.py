"""Ablation baselines over standardized technical features, exported as JSON."""

from collections import Counter, defaultdict

import numpy as np

from .features import DIM, SPECTRAL_DIM

VIEWS = {"spectrum": slice(0, SPECTRAL_DIM), "noise": slice(SPECTRAL_DIM, DIM),
         "combined": slice(0, DIM)}


def balanced_weights(rows):
    sources, groups = defaultdict(set), defaultdict(set)
    sizes = Counter()
    for row in rows:
        label, source, group = row["label"], row["source"], row["group"]
        sources[label].add(source)
        groups[label, source].add(group)
        sizes[label, source, group] += 1
    return np.asarray([.5 / len(sources[r["label"]]) / len(groups[r["label"], r["source"]])
                       / sizes[r["label"], r["source"], r["group"]] for r in rows])


def fit(vectors, labels, mode, weights=None):
    import torch

    x = np.asarray(vectors, dtype=np.float64)[:, VIEWS[mode]]
    y = np.asarray(labels, dtype=np.float64)
    if x.shape[0] != len(y) or set(y) != {0, 1} or not np.isfinite(x).all():
        raise ValueError("Training requires finite features from both classes")
    if weights is None:
        mean, scale = x.mean(axis=0), x.std(axis=0)
        weights = np.where(y == 1, .5 / y.sum(), .5 / (1-y).sum())
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != y.shape or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Invalid spectral sample weights")
        weights = weights / weights.sum()
        mean = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.where(scale > 1e-6, scale, 1.)
    features = torch.from_numpy((x - mean) / scale)
    target = torch.from_numpy(y)
    sample_weights = torch.from_numpy(weights)
    coefficients = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([coefficients, bias], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        losses = torch.nn.functional.binary_cross_entropy_with_logits(features @ coefficients + bias,
                                                                       target, reduction="none")
        loss = (losses * sample_weights).sum() + .01 * coefficients.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    result = dict(mode=mode, mean=mean.tolist(), scale=scale.tolist(),
                  weights=coefficients.detach().tolist(), bias=float(bias.detach()))
    score(result, vectors)
    return result


def score(model, vectors):
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != DIM:
        raise ValueError(f"Expected {DIM} spectral/noise features per video")
    x = x[:, VIEWS[model["mode"]]]
    logits = ((x - model["mean"]) / model["scale"]) @ model["weights"] + model["bias"]
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite spectral logits")
    return 1. / (1. + np.exp(-np.clip(logits, -60, 60)))


def scores_for(models, mode, rows, treatment="clean"):
    if mode in VIEWS:
        return score(models[mode], [r["vectors"][treatment] for r in rows])
    face = np.full(len(rows), np.nan)
    indices = [i for i, row in enumerate(rows) if treatment in row.get("face_vectors", {})]
    if indices:
        face[indices] = score(models["face"], [rows[i]["face_vectors"][treatment] for i in indices])
    if mode == "face":
        return face
    if mode != "full_face":
        raise ValueError(f"Unknown spectral variant: {mode}")
    full = scores_for(models, "combined", rows, treatment)
    weight = models["full_face"]["face_weight"]
    full[indices] = (1-weight)*full[indices] + weight*face[indices]
    return full
