import math

import numpy as np


def metrics(labels, scores, threshold=.5):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    if not len(y) or len(y) != len(p) or not np.isfinite(p).all():
        raise ValueError("Metrics require nonempty finite scores and matching labels")
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    # Average ranks give tied predictions (including a constant classifier) AUC .5.
    auc = None
    if tpr is not None and tnr is not None:
        order = np.argsort(p, kind="stable")
        ranks = np.empty(len(p), dtype=float)
        start = 0
        while start < len(p):
            end = start + 1
            while end < len(p) and p[order[end]] == p[order[start]]:
                end += 1
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
        n1, n0 = int(y.sum()), int((y == 0).sum())
        auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    precision = tp / (tp + fp) if tp + fp else 0.0
    average_precision = None
    if (y == 1).any():
        order = np.argsort(-p, kind="stable")
        sorted_scores, sorted_y = p[order], y[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(p)-1]
        cumulative = np.cumsum(sorted_y)[ends]
        recall = cumulative / y.sum()
        prec = cumulative / (ends + 1)
        average_precision = float(np.sum(np.diff(np.r_[0, recall]) * prec))
    brier = float(np.mean((p - y)**2))
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(n=len(y), tp=tp, tn=tn, fp=fp, fn=fn, accuracy=(tp + tn) / len(y),
                balanced_accuracy=(tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
                precision=precision, recall_fake=tpr, recall_real=tnr,
                false_positive_rate=1 - tnr if tnr is not None else None,
                f1=2*tp / max(2*tp + fp + fn, 1), auroc=auc, average_precision=average_precision, brier=brier,
                log_loss=float(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean()),
                threshold=float(threshold),
                score_quantiles={str(label): [float(v) for v in np.quantile(p[y == label], [0, .1, .5, .9, 1])]
                                 for label in (0, 1) if (y == label).any()})


def select_threshold(labels, scores, fpr=.05):
    # Select only on validation. >= threshold is the positive decision rule.
    real = np.sort(np.asarray(scores)[np.asarray(labels) == 0])[::-1]
    if not len(real):
        raise ValueError("Calibration needs real validation videos")
    allowed = int(math.floor(len(real) * fpr))
    return float(np.nextafter(real[min(allowed, len(real)-1)], np.inf))


def grouped_report(rows, threshold=.5):
    report = metrics([r["label"] for r in rows], [r["score"] for r in rows], threshold)
    report["by_source"] = {}
    for source in sorted({r["source"] for r in rows}):
        selected = [r for r in rows if r["source"] == source]
        report["by_source"][source] = metrics([r["label"] for r in selected], [r["score"] for r in selected], threshold)
    return report


def bootstrap_interval(rows, seed=42, repeats=300):
    """Resample origin groups rather than treating correlated frames as trials."""
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sample = [row for index in rng.integers(0, len(names), len(names)) for row in groups[names[index]]]
        y = np.asarray([r["label"] for r in sample])
        correct = (np.asarray([r["score"] for r in sample]) >= .5) == y
        if len(np.unique(y)) == 2:
            values.append(float((correct[y == 0].mean() + correct[y == 1].mean()) / 2))
    return [float(v) for v in np.quantile(values, [.025, .975])] if values else None
