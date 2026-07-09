import torch

from collections import defaultdict
from tqdm import tqdm

from training.config import VIDEO_TOP_K
from training.config import USE_AMP
from training.metrics import calculate_metrics


def _unpack_batch(batch):
    if len(batch) == 3:
        return batch

    images, labels = batch
    video_ids = [str(index) for index in range(len(labels))]
    return images, labels, video_ids


def _top_k_mean(values: list[float], k: int) -> float:
    top_values = sorted(values, reverse=True)[: max(1, min(k, len(values)))]
    return sum(top_values) / max(len(top_values), 1)


def _prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def validate(

    model,

    loader,

    criterion,

    device,

):

    model.eval()

    total_loss = 0

    predictions = []

    targets = []

    video_probs = defaultdict(list)

    video_targets = {}

    loop = tqdm(

        loader,

        desc="VALID",

        leave=False

    )

    use_amp = (
        USE_AMP
        and torch.cuda.is_available()
        and str(device).startswith("cuda")
    )

    with torch.inference_mode():

        for batch in loop:

            images, labels, video_ids = _unpack_batch(batch)

            images = images.to(device)

            labels = labels.to(device).unsqueeze(1)

            with torch.cuda.amp.autocast(
                enabled=use_amp,
            ):

                outputs = model(images)

                loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)

            preds = (probs > 0.5).cpu()

            predictions.append(preds)

            targets.append(labels.cpu())

            flat_probs = probs.detach().cpu().flatten().tolist()

            flat_labels = labels.detach().cpu().flatten().tolist()

            for video_id, probability, target in zip(
                video_ids,
                flat_probs,
                flat_labels,
            ):
                video_probs[str(video_id)].append(float(probability))
                video_targets[str(video_id)] = int(target)

            loop.set_postfix(

                loss=f"{loss.item():.4f}"

            )

    predictions = torch.cat(predictions)

    targets = torch.cat(targets)

    frame_metrics = calculate_metrics(

        predictions,

        targets

    )

    video_predictions = []

    video_labels = []

    for video_id, probabilities in video_probs.items():

        probability = _top_k_mean(
            probabilities,
            VIDEO_TOP_K,
        )

        video_predictions.append(
            probability > 0.5
        )

        video_labels.append(
            video_targets[video_id]
        )

    video_metrics = calculate_metrics(

        torch.tensor(video_predictions),

        torch.tensor(video_labels),

    )

    metrics = {
        **video_metrics,
        **_prefix_metrics(frame_metrics, "frame"),
        **_prefix_metrics(video_metrics, "video"),
    }

    metrics["loss"] = total_loss / len(loader)

    return metrics
