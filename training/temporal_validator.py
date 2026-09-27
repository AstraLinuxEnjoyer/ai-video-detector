from collections import defaultdict

import torch
from tqdm import tqdm

from training.metrics import calculate_metrics
from training.temporal_config import (
    TEMPORAL_USE_AMP,
    TEMPORAL_VIDEO_TOP_K,
)


def _top_k_mean(values: list[float], k: int) -> float:
    top_values = sorted(values, reverse=True)[: max(1, min(k, len(values)))]
    return sum(top_values) / max(len(top_values), 1)


def _prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def validate_temporal(
    model,
    loader,
    criterion,
    device,
) -> dict:
    model.eval()

    total_loss = 0.0
    clip_predictions = []
    clip_targets = []

    video_probs = defaultdict(list)
    video_targets = {}

    use_amp = (
        TEMPORAL_USE_AMP
        and torch.cuda.is_available()
        and str(device).startswith("cuda")
    )

    loop = tqdm(
        loader,
        desc="TEMPORAL VALID",
        leave=False,
    )

    with torch.inference_mode():
        for batch in loop:
            clips, labels, video_ids = batch

            clips = clips.to(device)
            labels = labels.to(device).unsqueeze(1)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp,
            ):
                outputs = model(clips)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).cpu()

            clip_predictions.append(preds)
            clip_targets.append(labels.cpu())

            flat_probs = probs.detach().cpu().flatten().tolist()
            flat_labels = labels.detach().cpu().flatten().tolist()

            for video_id, probability, target in zip(
                video_ids,
                flat_probs,
                flat_labels,
            ):
                video_probs[str(video_id)].append(float(probability))
                video_targets[str(video_id)] = int(target)

            running_clip_acc = (
                (torch.cat(clip_predictions).int() == torch.cat(clip_targets).int())
                .float()
                .mean()
                .item()
            )
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                clip_acc=f"{running_clip_acc:.4f}",
            )

    clip_predictions = torch.cat(clip_predictions)
    clip_targets = torch.cat(clip_targets)

    clip_metrics = calculate_metrics(clip_predictions, clip_targets)

    video_predictions = []
    video_labels = []
    video_scores = []

    for video_id, probabilities in video_probs.items():
        probability = _top_k_mean(probabilities, TEMPORAL_VIDEO_TOP_K)
        video_scores.append(probability)
        video_predictions.append(probability > 0.5)
        video_labels.append(video_targets[video_id])

    video_metrics = calculate_metrics(
        torch.tensor(video_predictions),
        torch.tensor(video_labels),
    )

    metrics = {
        **video_metrics,
        **_prefix_metrics(clip_metrics, "clip"),
        **_prefix_metrics(video_metrics, "video"),
    }
    metrics["loss"] = total_loss / max(len(loader), 1)
    metrics["videos"] = len(video_probs)
    metrics["clips"] = len(clip_predictions)
    metrics["mean_video_score"] = (
        sum(video_scores) / max(len(video_scores), 1)
    )

    return metrics
