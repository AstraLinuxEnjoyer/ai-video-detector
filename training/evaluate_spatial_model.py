import argparse
import csv
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from tqdm import tqdm

from training.config import (
    BEST_MODEL,
    FRAME_PAD_COLOR,
    IMAGE_SIZE,
    PRESERVE_ASPECT_RATIO,
    ROOT,
)

try:
    cv2.setLogLevel(0)
except Exception:
    pass


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class VideoSample:
    path: Path
    label: int
    label_name: str
    source: str


@dataclass
class Prediction:
    path: Path
    label: int
    label_name: str
    source: str
    probability: float
    mean_probability: float
    max_probability: float
    top_k_probability: float
    prediction: int
    frame_count: int


def build_eval_model(model_path: Path, device: torch.device):
    model = efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        1,
    )

    state = torch.load(
        model_path,
        map_location=device,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def scan_test_videos(root: Path) -> list[VideoSample]:
    samples: list[VideoSample] = []
    classes = {
        "real": 0,
        "fake": 1,
    }

    for label_name, label in classes.items():
        class_dir = root / label_name
        if not class_dir.exists():
            continue

        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            relative = path.relative_to(class_dir)
            source = relative.parts[0] if len(relative.parts) > 1 else label_name

            samples.append(
                VideoSample(
                    path=path,
                    label=label,
                    label_name=label_name,
                    source=source,
                )
            )

    return samples


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return max(frame_count, 1)


def decode_frame(frame: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if PRESERVE_ASPECT_RATIO:
        height, width = frame.shape[:2]
        scale = image_size / max(height, width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))

        frame = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        canvas = np.full(
            (image_size, image_size, 3),
            FRAME_PAD_COLOR,
            dtype=np.uint8,
        )
        y = (image_size - resized_height) // 2
        x = (image_size - resized_width) // 2
        canvas[y:y + resized_height, x:x + resized_width] = frame
        return canvas

    frame = cv2.resize(
        frame,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    return frame


def read_frame(video_path: Path, frame_number: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    frame_number = max(0, min(frame_number, frame_count - 1))

    candidates = [frame_number]
    for offset in range(1, min(8, frame_count)):
        previous_frame = frame_number - offset
        next_frame = frame_number + offset
        if previous_frame >= 0:
            candidates.append(previous_frame)
        if next_frame < frame_count:
            candidates.append(next_frame)

    for candidate in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        success, frame = cap.read()
        if success and frame is not None:
            cap.release()
            return decode_frame(frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        success, frame = cap.read()
        if success and frame is not None:
            cap.release()
            return decode_frame(frame)

        position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if position >= frame_count:
            break

    cap.release()
    raise RuntimeError(f"Cannot read frame from {video_path}")


def sampled_frame_numbers(video_path: Path, samples: int) -> list[int]:
    frame_count = get_frame_count(video_path)
    if frame_count <= 1:
        return [0]

    samples = max(1, min(samples, frame_count))
    if samples == 1:
        return [0]

    return [
        round(index * (frame_count - 1) / (samples - 1))
        for index in range(samples)
    ]


def predict_frames(
    model,
    transform,
    device: torch.device,
    frames: list[np.ndarray],
    batch_size: int,
) -> list[float]:
    probabilities: list[float] = []

    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]
            tensors = [
                transform(Image.fromarray(frame))
                for frame in batch_frames
            ]
            batch = torch.stack(tensors).to(device)
            outputs = model(batch)
            batch_probs = torch.sigmoid(outputs).detach().cpu().flatten().tolist()
            probabilities.extend(float(item) for item in batch_probs)

    return probabilities


def aggregate_probability(probabilities: list[float], top_k: int) -> tuple[float, float, float]:
    mean_probability = sum(probabilities) / max(len(probabilities), 1)
    max_probability = max(probabilities) if probabilities else 0.0
    top = sorted(probabilities, reverse=True)[: max(1, min(top_k, len(probabilities)))]
    top_k_probability = sum(top) / max(len(top), 1)

    return mean_probability, max_probability, top_k_probability


def calculate_metrics(predictions: list[Prediction]) -> dict:
    tp = sum(1 for item in predictions if item.prediction == 1 and item.label == 1)
    tn = sum(1 for item in predictions if item.prediction == 0 and item.label == 0)
    fp = sum(1 for item in predictions if item.prediction == 1 and item.label == 0)
    fn = sum(1 for item in predictions if item.prediction == 0 and item.label == 1)

    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "total": total,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def grouped_metrics(predictions: list[Prediction], key_fn):
    groups: dict[str, list[Prediction]] = {}
    for item in predictions:
        groups.setdefault(key_fn(item), []).append(item)

    return {
        group: calculate_metrics(items)
        for group, items in sorted(groups.items())
    }


def setup_report_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        report_dir = output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = ROOT / "reports" / f"spatial_eval_{timestamp}"

    (report_dir / "heatmaps").mkdir(parents=True, exist_ok=True)
    (report_dir / "compressed").mkdir(parents=True, exist_ok=True)

    return report_dir


def setup_logger(report_dir: Path):
    log_path = report_dir / "evaluation.log"

    logger = logging.getLogger("spatial_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def write_predictions_csv(path: Path, predictions: list[Prediction]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "path",
            "label",
            "source",
            "probability",
            "mean_probability",
            "max_probability",
            "top_k_probability",
            "prediction",
            "frame_count",
            "correct",
        ])
        for item in predictions:
            writer.writerow([
                str(item.path),
                item.label_name,
                item.source,
                f"{item.probability:.6f}",
                f"{item.mean_probability:.6f}",
                f"{item.max_probability:.6f}",
                f"{item.top_k_probability:.6f}",
                "fake" if item.prediction == 1 else "real",
                item.frame_count,
                int(item.prediction == item.label),
            ])


def write_metrics_csv(path: Path, metrics_by_group: dict[str, dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "group",
            "total",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tp",
            "tn",
            "fp",
            "fn",
        ])
        for group, metrics in metrics_by_group.items():
            writer.writerow([
                group,
                metrics["total"],
                f"{metrics['accuracy']:.6f}",
                f"{metrics['precision']:.6f}",
                f"{metrics['recall']:.6f}",
                f"{metrics['f1']:.6f}",
                metrics["tp"],
                metrics["tn"],
                metrics["fp"],
                metrics["fn"],
            ])


def sanitize_name(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe)[:160]


class GradCam:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_handle = model.features[-1].register_forward_hook(
            self._save_activations
        )
        self.backward_handle = model.features[-1].register_full_backward_hook(
            self._save_gradients
        )

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def make_heatmap(self, tensor: torch.Tensor, target_fake: bool) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        output = self.model(tensor)
        logit = output[:, 0]
        score = logit if target_fake else -logit
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cv2.resize(
            cam,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )

        return cam


def save_heatmap(
    model,
    transform,
    device: torch.device,
    sample: VideoSample,
    frame_number: int,
    probability: float,
    prediction: int,
    output_path: Path,
) -> None:
    frame = read_frame(sample.path, frame_number)
    image = Image.fromarray(frame)
    tensor = transform(image).unsqueeze(0).to(device)

    grad_cam = GradCam(model)
    try:
        heatmap = grad_cam.make_heatmap(
            tensor,
            target_fake=prediction == 1,
        )
    finally:
        grad_cam.close()

    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(frame_bgr, 0.55, colored, 0.45, 0)

    caption = (
        f"label={sample.label_name} "
        f"pred={'fake' if prediction == 1 else 'real'} "
        f"p={probability:.3f}"
    )
    cv2.putText(
        overlay,
        caption,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), overlay)


def evaluate_video(
    model,
    transform,
    device: torch.device,
    sample: VideoSample,
    frames_per_video: int,
    top_k: int,
    threshold: float,
    batch_size: int,
) -> tuple[Prediction, list[tuple[int, float]]]:
    frame_numbers = sampled_frame_numbers(sample.path, frames_per_video)
    frames = [
        read_frame(sample.path, frame_number)
        for frame_number in frame_numbers
    ]
    probabilities = predict_frames(
        model,
        transform,
        device,
        frames,
        batch_size,
    )
    mean_probability, max_probability, top_k_probability = aggregate_probability(
        probabilities,
        top_k,
    )
    prediction = int(top_k_probability >= threshold)

    return (
        Prediction(
            path=sample.path,
            label=sample.label,
            label_name=sample.label_name,
            source=sample.source,
            probability=top_k_probability,
            mean_probability=mean_probability,
            max_probability=max_probability,
            top_k_probability=top_k_probability,
            prediction=prediction,
            frame_count=len(frame_numbers),
        ),
        list(zip(frame_numbers, probabilities)),
    )


def log_metrics(logger, title: str, metrics: dict) -> None:
    logger.info(
        "%s | total=%s accuracy=%.4f precision=%.4f recall=%.4f "
        "f1=%.4f tp=%s tn=%s fp=%s fn=%s",
        title,
        metrics["total"],
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["tp"],
        metrics["tn"],
        metrics["fp"],
        metrics["fn"],
    )


def pick_heatmap_targets(
    predictions: list[Prediction],
    frame_probs: dict[Path, list[tuple[int, float]]],
    limit: int,
) -> list[tuple[Prediction, int, float]]:
    if limit <= 0:
        return []

    errors = [item for item in predictions if item.prediction != item.label]
    correct = [item for item in predictions if item.prediction == item.label]

    errors.sort(key=lambda item: abs(item.probability - 0.5), reverse=True)
    correct.sort(key=lambda item: abs(item.probability - 0.5), reverse=True)

    ordered = [*errors, *correct]
    targets = []

    for prediction in ordered[:limit]:
        frames = frame_probs[prediction.path]
        if prediction.prediction == 1:
            frame_number, probability = max(frames, key=lambda item: item[1])
        else:
            frame_number, probability = min(frames, key=lambda item: item[1])

        targets.append((prediction, frame_number, probability))

    return targets


def compress_video(
    input_path: Path,
    output_path: Path,
    crf: int,
    scale_width: int,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vf",
        f"scale={scale_width}:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-an",
        str(output_path),
    ]

    result = subprocess.run(command)
    return result.returncode == 0 and output_path.exists()


def run_compression_test(
    logger,
    model,
    transform,
    device: torch.device,
    samples: list[VideoSample],
    original_predictions: dict[Path, Prediction],
    report_dir: Path,
    frames_per_video: int,
    top_k: int,
    threshold: float,
    batch_size: int,
    sample_limit: int,
    crf: int,
    scale_width: int,
) -> list[dict]:
    if sample_limit <= 0:
        return []

    selected = samples[:sample_limit]
    rows = []

    logger.info("Compression test videos: %s", len(selected))

    for sample in tqdm(selected, desc="COMPRESSION"):
        compressed_path = (
            report_dir
            / "compressed"
            / f"{sanitize_name(sample.path.stem)}_crf{crf}.mp4"
        )
        ok = compress_video(
            sample.path,
            compressed_path,
            crf=crf,
            scale_width=scale_width,
        )
        if not ok:
            logger.warning("Compression failed or ffmpeg missing: %s", sample.path)
            continue

        compressed_sample = VideoSample(
            path=compressed_path,
            label=sample.label,
            label_name=sample.label_name,
            source=f"{sample.source}:compressed",
        )
        prediction, _frame_probs = evaluate_video(
            model,
            transform,
            device,
            compressed_sample,
            frames_per_video,
            top_k,
            threshold,
            batch_size,
        )
        original = original_predictions[sample.path]

        rows.append({
            "path": str(sample.path),
            "label": sample.label_name,
            "original_probability": original.probability,
            "compressed_probability": prediction.probability,
            "delta": prediction.probability - original.probability,
            "original_prediction": original.prediction,
            "compressed_prediction": prediction.prediction,
            "changed": int(original.prediction != prediction.prediction),
        })

    return rows


def select_balanced_samples(
    predictions: list[Prediction],
    limit: int,
) -> list[VideoSample]:
    if limit <= 0:
        return []

    real = [item for item in predictions if item.label == 0]
    fake = [item for item in predictions if item.label == 1]

    per_class = max(1, limit // 2)
    selected = [*real[:per_class], *fake[:per_class]]

    if len(selected) < limit:
        used = {item.path for item in selected}
        remaining = [
            item
            for item in predictions
            if item.path not in used
        ]
        selected.extend(remaining[: limit - len(selected)])

    return [
        VideoSample(
            path=item.path,
            label=item.label,
            label_name=item.label_name,
            source=item.source,
        )
        for item in selected[:limit]
    ]


def write_compression_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "path",
            "label",
            "original_probability",
            "compressed_probability",
            "delta",
            "original_prediction",
            "compressed_prediction",
            "changed",
        ])
        for row in rows:
            writer.writerow([
                row["path"],
                row["label"],
                f"{row['original_probability']:.6f}",
                f"{row['compressed_probability']:.6f}",
                f"{row['delta']:.6f}",
                "fake" if row["original_prediction"] == 1 else "real",
                "fake" if row["compressed_prediction"] == 1 else "real",
                row["changed"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "test_videos"))
    parser.add_argument("--model", default=str(BEST_MODEL))
    parser.add_argument("--output", default=None)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--heatmaps", type=int, default=12)
    parser.add_argument("--compression-samples", type=int, default=10)
    parser.add_argument("--compression-crf", type=int, default=32)
    parser.add_argument("--compression-width", type=int, default=1280)
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument("--no-compression", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input)
    model_path = Path(args.model)
    output_dir = Path(args.output) if args.output else None
    report_dir = setup_report_dir(output_dir)
    logger = setup_logger(report_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Test video folder not found: {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Spatial model evaluation")
    logger.info("Input     : %s", input_dir)
    logger.info("Model     : %s", model_path)
    logger.info("Report dir: %s", report_dir)
    logger.info("Device    : %s", device)
    logger.info("Frames/video=%s top_k=%s threshold=%.3f", args.frames, args.top_k, args.threshold)

    samples = scan_test_videos(input_dir)
    logger.info("Found test videos: %s", len(samples))
    logger.info("Real videos: %s", sum(1 for item in samples if item.label == 0))
    logger.info("Fake videos: %s", sum(1 for item in samples if item.label == 1))

    if not samples:
        raise RuntimeError("No test videos found.")

    model = build_eval_model(model_path, device)
    transform = build_transform()

    predictions: list[Prediction] = []
    frame_probs: dict[Path, list[tuple[int, float]]] = {}

    for sample in tqdm(samples, desc="EVALUATE"):
        try:
            prediction, video_frame_probs = evaluate_video(
                model,
                transform,
                device,
                sample,
                frames_per_video=args.frames,
                top_k=args.top_k,
                threshold=args.threshold,
                batch_size=args.batch_size,
            )
        except Exception as exc:
            logger.exception("Failed to evaluate %s: %s", sample.path, exc)
            continue

        predictions.append(prediction)
        frame_probs[sample.path] = video_frame_probs

    overall = calculate_metrics(predictions)
    log_metrics(logger, "Overall", overall)

    by_label = grouped_metrics(predictions, lambda item: item.label_name)
    for group, metrics in by_label.items():
        log_metrics(logger, f"Label={group}", metrics)

    by_source = grouped_metrics(
        predictions,
        lambda item: f"{item.label_name}/{item.source}",
    )
    for group, metrics in by_source.items():
        log_metrics(logger, f"Source={group}", metrics)

    write_predictions_csv(report_dir / "predictions.csv", predictions)
    write_metrics_csv(report_dir / "metrics_by_source.csv", by_source)

    if not args.no_heatmaps:
        targets = pick_heatmap_targets(
            predictions,
            frame_probs,
            limit=args.heatmaps,
        )
        logger.info("Saving heatmaps: %s", len(targets))
        for index, (prediction, frame_number, probability) in enumerate(targets, start=1):
            sample = VideoSample(
                path=prediction.path,
                label=prediction.label,
                label_name=prediction.label_name,
                source=prediction.source,
            )
            output_path = (
                report_dir
                / "heatmaps"
                / f"{index:03d}_{prediction.label_name}_{sanitize_name(prediction.path.stem)}.jpg"
            )
            save_heatmap(
                model,
                transform,
                device,
                sample,
                frame_number,
                probability,
                prediction.prediction,
                output_path,
            )

    if not args.no_compression:
        original_predictions = {
            item.path: item
            for item in predictions
        }
        compression_samples = select_balanced_samples(
            predictions,
            args.compression_samples,
        )
        compression_rows = run_compression_test(
            logger,
            model,
            transform,
            device,
            compression_samples,
            original_predictions,
            report_dir,
            frames_per_video=args.frames,
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            sample_limit=args.compression_samples,
            crf=args.compression_crf,
            scale_width=args.compression_width,
        )
        write_compression_csv(
            report_dir / "compression_test.csv",
            compression_rows,
        )
        changed = sum(row["changed"] for row in compression_rows)
        total = max(len(compression_rows), 1)
        logger.info(
            "Compression changed predictions: %s/%s (%.2f%%)",
            changed,
            len(compression_rows),
            100 * changed / total,
        )

    logger.info("Done.")
    logger.info("Main log: %s", report_dir / "evaluation.log")
    logger.info("Predictions CSV: %s", report_dir / "predictions.csv")
    logger.info("Source metrics CSV: %s", report_dir / "metrics_by_source.csv")


if __name__ == "__main__":
    main()
