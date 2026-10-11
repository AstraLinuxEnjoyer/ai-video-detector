"""Bounded train-only hard-example weights and paired patch loss."""
import math

import numpy as np
import torch
import torch.nn.functional as F


def hard_weights(ids, sources, history):
    pools = {}
    for entry in history.values():
        pools.setdefault(entry["source"], []).append(entry["loss"])
    medians = {source: max(float(np.median(values)), .1) for source, values in pools.items()}
    return [min(2., max(1., history[item]["loss"] / medians[source]))
            if item in history and source in medians else 1. for item, source in zip(ids, sources)]


def update_history(history, ids, sources, losses):
    for item, source, loss in zip(ids, sources, losses):
        if not math.isfinite(loss):
            raise ValueError("Nonfinite hard-example loss")
        old = history.get(item, {}).get("loss", loss)
        history[item] = dict(source=source, loss=.8*old + .2*loss)


def paired_loss(logits, labels, weights, consistency=.5, hybrid=False):
    # B x 2 views x 2 patches. Small smoothing discourages saturation, not calibration.
    targets = labels[:, None, None].expand_as(logits)
    supervised = F.binary_cross_entropy_with_logits(logits.float(), targets*.98 + .01, reduction="none")
    from .patches import aggregate_views
    raw = aggregate_views(F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none").mean(1), hybrid)
    agreement = aggregate_views((logits[:, 0].float().sigmoid() - logits[:, 1].float().sigmoid()).square(), hybrid)
    per_frame = aggregate_views(supervised.mean(1), hybrid) + consistency * agreement
    loss = (per_frame * weights).sum() / weights.sum()
    return loss, raw.detach(), agreement.detach()


def source_score(report, minimum=10):
    """Each sufficiently populated source within each class gets equal weight."""
    groups = {0: [], 1: []}
    excluded = []
    for source, row in report["by_source"].items():
        for label, correct, count in ((0, row["tn"], row["tn"]+row["fp"]),
                                      (1, row["tp"], row["tp"]+row["fn"])):
            if count >= minimum:
                groups[label].append(correct/count)
            elif count:
                excluded.append(dict(source=source, label=label, n=count))
    if not all(groups.values()):
        raise ValueError("Source selection metric needs sufficiently populated sources in both classes")
    means = {str(label): float(np.mean(values)) for label, values in groups.items()}
    return dict(score=(means["0"]+means["1"])/2, by_class=means, excluded_small_groups=excluded,
                worst_source_accuracy=min(value for values in groups.values() for value in values))


def selection_score(clean, stress, minimum=10):
    a, b = source_score(clean, minimum), source_score(stress, minimum)
    return dict(score=(a["score"]+b["score"])/2, clean=a, stress=b)
