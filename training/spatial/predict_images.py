"""Unlabelled photo inference using the best spatial_v2 weights; no training."""
import argparse
import hashlib
import io
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .common import ROOT, atomic_json
from .engine import Frames, build_model

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def image_tensor(path, image_size):
    with Image.open(path) as original:
        if getattr(original, "n_frames", 1) != 1:
            raise ValueError("Animated or multipage image: provide one still image")
        image = ImageOps.exif_transpose(original)
        if "A" in image.getbands() or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        # Match the video's bounded decode size, then reuse its evaluation transforms.
        image = ImageOps.contain(image, (640, 640), method=Image.Resampling.BICUBIC)
        transforms = Frames([], image_size)
        return transforms.normalize(transforms.eval_crop(image))


def best_snapshot(run):
    weights_path, meta_path = run / "best.pt", run / "best.json"
    if not weights_path.exists() or not meta_path.exists():
        raise ValueError("No best model yet. Complete at least one epoch with validation first")
    for _ in range(5):
        try:
            before = (weights_path.stat(), meta_path.stat())
            if before[0].st_mtime_ns > before[1].st_mtime_ns:
                time.sleep(.2)
                continue
            metadata_text = meta_path.read_text(encoding="utf-8")
            content = weights_path.read_bytes()
            after = (weights_path.stat(), meta_path.stat())
            if [(s.st_size, s.st_mtime_ns) for s in before] == [(s.st_size, s.st_mtime_ns) for s in after]:
                return (torch.load(io.BytesIO(content), map_location="cpu", weights_only=True),
                        json.loads(metadata_text), hashlib.sha256(content).hexdigest())
        except (PermissionError, FileNotFoundError):
            pass
        time.sleep(.2)
    raise RuntimeError("Best model is being replaced. Wait for checkpoint saving to finish and retry")


def collect_images(input_path):
    if input_path.is_file():
        if input_path.suffix.lower() not in EXTENSIONS:
            raise ValueError("Unsupported image format. Use JPG, PNG, WEBP, BMP or TIFF")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input folder does not exist: {input_path}")
    return sorted(p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def classify(model, path, size, device, threshold=.5):
    tensor = image_tensor(path, size).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor).float().reshape(-1)
        if logits.numel() != 1:
            raise ValueError("Expected one binary output for one photo")
        score = float(logits.sigmoid().item())
    if not math.isfinite(score):
        raise RuntimeError("Nonfinite model output")
    return dict(path=str(path.resolve()), prediction="FAKE" if score >= threshold else "REAL",
                fake_score=score, threshold=threshold, near_threshold=abs(score-threshold) < .05)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify unlabelled photos using the best spatial model")
    parser.add_argument("--input", type=Path, default=ROOT / "photo_check")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threshold", type=float, default=.5)
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    paths = collect_images(args.input)
    if not paths:
        print(f"No photos yet. Put JPG/PNG/WEBP/BMP/TIFF files here: {args.input.resolve()}")
        return
    latest = ROOT / "cache/spatial_v2/latest.json"
    if not args.run_dir and not latest.exists():
        raise ValueError("No trained spatial_v2 run found; specify --run-dir")
    run = (args.run_dir or Path(json.loads(latest.read_text())["run_dir"])).resolve()
    weights, best, sha = best_snapshot(run)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use --device cpu")
    model = build_model(False)
    model.load_state_dict(weights)
    del weights
    model.to(device).eval()
    # Avoid starving an ongoing training decoder on a many-core machine.
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Model: {run / 'best.pt'} | epoch={best['epoch']} | device={device}")
    print(f"Photos: {len(paths)} | threshold={args.threshold:.4f}")
    print("REAL = camera/real prediction; FAKE = generated prediction. Labels are NOT read from folders.")
    print("fake_score is a model score, NOT a calibrated probability. This is a single-photo test, not a video test.")
    rows, errors = [], []
    for path in paths:
        try:
            row = classify(model, path, best["image_size"], device, args.threshold)
            rows.append(row)
            note = " | NEAR THRESHOLD" if row["near_threshold"] else ""
            print(f"{row['prediction']:4} | {path.name} | fake_score={row['fake_score']:.6f}{note}", flush=True)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(dict(path=str(path.resolve()), error=str(exc)))
            print(f"ERROR | {path.name} | {exc}", flush=True)
    report = ROOT / "reports" / f"photo_check_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    atomic_json(report, dict(run_dir=str(run), weights_sha256=sha, epoch=best["epoch"], device=str(device),
                threshold=args.threshold, video_validation_threshold=best.get("threshold"),
                caveat="Unlabelled single photos; no accuracy calculated; scores are not calibrated probabilities",
                predictions=rows, errors=errors))
    print(f"Done: {len(rows)} predictions, {len(errors)} errors. Report: {report}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
