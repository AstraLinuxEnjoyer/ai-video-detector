"""Evaluate a v2 run on independently collected real/fake folders."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch

from .common import ROOT, atomic_json, logger_for
from .data import EXTENSIONS
from .engine import build_model, evaluate, write_predictions
from .media import FrameCache, prepare_frames
from .metrics import grouped_report


def external_evaluation(model, input_root, output, args, device, logger, train_ids, threshold):
    records = []
    seen_hashes = {}
    for label, folder in enumerate(("real", "fake")):
        for path in sorted((input_root / folder).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha in seen_hashes:
                if seen_hashes[sha] != label:
                    raise ValueError(f"Conflicting real/fake labels in external test: {path}")
                continue
            seen_hashes[sha] = label
            relative = path.relative_to(input_root / folder)
            source = relative.parts[0] if len(relative.parts) > 1 else folder
            stat = path.stat()
            records.append(dict(id=sha, label=label, group=sha, source=source, path=str(path.resolve()),
                                size=stat.st_size, mtime=stat.st_mtime_ns))
    if not records:
        report = dict(status="no_data", input=str(input_root))
        logger.warning("No external videos in %s/real and %s/fake", input_root, input_root)
        atomic_json(output / "external_report.json", report)
        return report
    cache = FrameCache(output, args.cache_gb, args.reserve_gb)
    prepared, failures = prepare_frames(records, cache, "external", args.val_frames,
                                         args.seed, logger, output, args.decode_workers)
    if failures:
        report = dict(status="incomplete", requested=len(records), failures=failures)
    else:
        rows, report = evaluate(model, prepared, args, device, logger, "EXTERNAL")
        write_predictions(output / "external_predictions.csv", rows)
        report["calibrated_on_validation"] = grouped_report(rows, threshold)
        report["exact_train_overlaps"] = [r["id"] for r in rows if r["id"] in train_ids]
        report["independence"] = "not_independent" if report["exact_train_overlaps"] else "No exact train matches; provenance and near-duplicates not verified"
        if len(rows) < 100:
            logger.warning("Only %d external videos: treat this as diagnostics, not a reliable population accuracy estimate", len(rows))
    atomic_json(output / "external_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "test_videos")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    latest = ROOT / "cache/spatial_v2/latest.json"
    if not args.run_dir and not latest.exists():
        raise ValueError("No spatial_v2 run found. Train first or specify --run-dir")
    run = args.run_dir or Path(json.loads(latest.read_text())["run_dir"])
    if not (run / "best.pt").exists() or not (run / "best.json").exists():
        raise ValueError("No best model yet. Complete at least one training epoch and validation first")
    config = argparse.Namespace(**json.loads((run / "config.json").read_text()))
    best = json.loads((run / "best.json").read_text())
    output = run / "external" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    logger = logger_for(output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(False).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    snapshot = json.loads((run / "splits.json").read_text())
    train_ids = {r["id"] for r in snapshot["splits"]["train"]}
    external_evaluation(model, args.input, output, config, device, logger, train_ids, best["threshold"])
    logger.info("External report: %s", output)


if __name__ == "__main__":
    main()
