import torch


def calculate_metrics(predictions, targets):

    predictions = predictions.int()
    targets = targets.int()

    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    precision = tp / max(tp + fp, 1)

    recall = tp / max(tp + fn, 1)

    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "tp": tp,

        "tn": tn,

        "fp": fp,

        "fn": fn,

    }