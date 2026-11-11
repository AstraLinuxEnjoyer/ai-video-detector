"""Isolated artifact-oriented experiment; shares the v2 lock and frozen partitions."""
import argparse
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path

import torch
from filelock import FileLock, Timeout

from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.spatial.data import assert_disjoint, fixed_selection, EXTENSIONS
from training.spatial.engine import atomic_torch, build_model
from training.spatial.lineage import load_parent, file_sha
from training.spatial.media import FrameCache
from training.spatial.predict_images import best_snapshot, collect_images
from .engine import train, evaluate, patch_scores, write_predictions
from .media import prepare
from .patches import load_image, view_tensor, aggregate_views
from .sampling import MatchedRotation, distribution
from .metadata import audit_dimensions, attach_dimensions

DEFAULTS = dict(epochs=12, videos_per_epoch=2000, frames_per_video=2, val_frames=4,
                val_videos=800, test_videos=800, batch_size=4, image_size=224,
                seed=42, workers=2, decode_workers=2, patience=5, min_source_count=10,
                backbone_lr=5e-6, head_lr=2e-5, cache_gb=20., reserve_gb=10.,
                save_every=50, log_every=20, device="cuda")
RECIPES = ("patch-v1", "hybrid-v2", "full-frame-v1")
MATCHED_RECIPES = {"hybrid-v2", "full-frame-v1"}


def initialize(run, source, given):
    path = run / "config.json"
    if path.exists():
        config = json.loads(path.read_text())
        if any(config[key] != value for key, value in given.items()):
            raise ValueError("Run settings are frozen. Choose a new --run-dir to change the recipe")
        if source and str(source.resolve()) != config["source_run"]:
            raise ValueError("Cannot change parent of an existing experiment")
        snapshot = json.loads((run / "splits.json").read_text(encoding="utf-8"))
        if fingerprint(snapshot["splits"]) != snapshot["fingerprint"]:
            raise ValueError("Corrupt snapshot")
        assert_disjoint(snapshot["splits"])
        lineage = json.loads((run / "lineage.json").read_text())
        if file_sha(run / "initial.pt") != lineage["initial_weights_sha256"]:
            raise ValueError("Initial weight snapshot changed")
        if config.get("recipe") in MATCHED_RECIPES and file_sha(run / "dimensions.json") != config["dimensions_sha256"]:
            raise ValueError("Frozen dimension audit changed; use a new run")
        return argparse.Namespace(**config), snapshot
    source = source or Path(json.loads((ROOT / "cache/spatial_v2/latest.json").read_text())["run_dir"])
    if source.resolve() == run.resolve():
        raise ValueError("Source and new run must be different directories")
    config = {**DEFAULTS, "recipe": "patch-v1", **given, "source_run": str(source.resolve())}
    if config["recipe"] not in RECIPES:
        raise ValueError("Unknown recipe")
    for key in ("epochs", "videos_per_epoch", "frames_per_video", "val_frames", "val_videos", "test_videos",
                "batch_size", "image_size", "decode_workers", "patience", "min_source_count", "save_every", "log_every",
                "backbone_lr", "head_lr", "cache_gb", "reserve_gb"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if config["workers"] < 0 or any(config[k] < 2 for k in ("videos_per_epoch", "val_videos", "test_videos")):
        raise ValueError("Invalid worker or sample count")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    lineage, splits = load_parent(source)
    assert_disjoint(splits)
    if not set(lineage["seen_ids"]) <= {r["id"] for r in splits["train"]}:
        raise ValueError("Parent exposure history contains IDs outside train")
    if config["recipe"] in MATCHED_RECIPES:
        parent_config = json.loads((source / "config.json").read_text()) if (source / "config.json").exists() else {}
        parent_dimensions = source / "dimensions.json"
        if config["recipe"] == "full-frame-v1" and parent_config.get("dimensions_sha256"):
            if file_sha(parent_dimensions) != parent_config["dimensions_sha256"]:
                raise ValueError("Parent dimension audit changed")
            dimensions = json.loads(parent_dimensions.read_text())
            atomic_json(run / "dimensions.json", dimensions)
        else:
            dimensions = audit_dimensions(splits["train"], run, logging.getLogger("spatial_v2"),
                                          max(4, config["decode_workers"]*2))
        config["dimensions_sha256"] = file_sha(run / "dimensions.json")
        sampler = MatchedRotation(attach_dimensions(splits["train"], dimensions), config["seed"], previously_seen=lineage["seen_ids"])
        preview = sampler.take(config["videos_per_epoch"])
        atomic_json(run / "sampling_preview.json", dict(sampler.audit,
                    selected=distribution(preview), sources={s: sum(r["source"] == s for r in preview)
                    for s in sorted({r["source"] for r in preview})}))
    if shutil.disk_usage(run).free < (config["reserve_gb"]+1)*1024**3:
        raise OSError("Insufficient disk space for snapshot/checkpoints and reserve")
    weights, best, sha = best_snapshot(source)
    if sha != lineage["weights_sha256"]:
        raise RuntimeError("Parent weights changed while preparing experiment")
    atomic_torch(run / "initial.pt", weights)
    lineage["initial_weights_sha256"] = file_sha(run / "initial.pt")
    payload = dict(schema=1, fingerprint=fingerprint(splits), splits=splits)
    atomic_json(run / "splits.json", payload)
    atomic_json(run / "lineage.json", lineage)
    atomic_json(run / "parent_best.json", best)
    # Config is the completion marker, written after all preparation artifacts.
    atomic_json(path, config)
    return argparse.Namespace(**config), payload


def final_test(run, payload, args, log):
    if (run / "final_report.json").exists():
        return
    device = torch.device(args.device)
    model = build_model(False).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    cache = FrameCache(run, args.cache_gb, args.reserve_gb)
    reports = dict(caveat="Old tests were inspected previously; not a fresh blind benchmark. OOD may be fake-only.")
    for split in ("test", "ood"):
        chosen = fixed_selection(payload["splits"][split], args.test_videos, args.seed)
        if not chosen:
            reports[split] = dict(status="no_data")
            continue
        records = prepare(chosen, cache, split, args.val_frames, args.seed, log, run, args.decode_workers)
        rows, report = evaluate(model, records, args, device, log, split)
        write_predictions(run / f"{split}_predictions.csv", rows)
        reports[split] = report
    atomic_json(run / "final_report.json", reports)


def inference(run, args, input_path, photos):
    weights, best, sha = best_snapshot(run)
    if best.get("recipe") != args.recipe or args.recipe not in RECIPES:
        raise ValueError("Weights and inference recipe disagree")
    hybrid = args.recipe == "hybrid-v2"
    model = build_model(False).cpu().eval()
    model.load_state_dict(weights)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = run / "evaluations" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    log = logger_for(output)
    if photos:
        rows, errors = [], []
        for path in collect_images(input_path):
            try:
                image = load_image(path)
                with torch.inference_mode():
                    views = view_tensor(image, args.image_size, hybrid=hybrid, full_frame=args.recipe == "full-frame-v1")
                    patches = patch_scores(model, views[None], torch.device("cpu"))[0]
                score = float(aggregate_views(patches, hybrid))
                row = dict(path=str(path), prediction="FAKE" if score >= .5 else "REAL", score=score,
                           patch_scores=patches.tolist(), patch_spread=float(patches.max()-patches.min()))
                if hybrid:
                    row.update(global_score=float(patches[0]), local_score=float(patches[1:].mean()))
                rows.append(row)
                log.info("%s | %s | score=%.5f patch_spread=%.5f (uncalibrated)", row["prediction"], path.name, score, row["patch_spread"])
            except (OSError, ValueError, RuntimeError) as exc:
                errors.append(dict(path=str(path), error=str(exc)))
                log.error("Photo failed %s: %s", path, exc)
        atomic_json(output / "photos.json", dict(weights_sha256=sha, epoch=best["epoch"], predictions=rows, errors=errors,
                    caveat="Scores are not calibrated probabilities. Patch disagreement does not detect every error."))
        if errors:
            raise RuntimeError("Some photos failed; see photos.json")
        return
    records, seen = [], {}
    payload = json.loads((run / "splits.json").read_text(encoding="utf-8"))
    train_ids = {r["id"] for r in payload["splits"]["train"]}
    for label, folder in enumerate(("real", "fake")):
        for path in sorted((input_path / folder).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            sha = file_sha(path)
            if sha in train_ids:
                raise ValueError(f"External test overlaps training: {path}")
            if sha in seen:
                if seen[sha] != label:
                    raise ValueError("Conflicting external labels")
                continue
            seen[sha] = label
            stat = path.stat()
            relative = path.relative_to(input_path / folder)
            records.append(dict(id=sha, group=sha, label=label, source=relative.parts[0] if len(relative.parts)>1 else folder,
                                path=str(path.resolve()), size=stat.st_size, mtime=stat.st_mtime_ns))
    if not records:
        raise ValueError("No videos in real/fake subdirectories")
    prepared = prepare(records, FrameCache(output, args.cache_gb, args.reserve_gb), "external",
                       args.val_frames, args.seed, log, output, args.decode_workers)
    rows, report = evaluate(model, prepared, args, torch.device("cpu"), log, "EXTERNAL")
    report["caveat"] = "No exact train matches; provenance/near duplicates require review. Not used for checkpoint selection."
    write_predictions(output / "predictions.csv", rows)
    atomic_json(output / "report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "cache/artifact_spatial/runs/patch_v1")
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--recipe", choices=RECIPES, default=argparse.SUPPRESS)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--predict-images", action="store_true")
    modes.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--input", type=Path)
    for name, default in DEFAULTS.items():
        parser.add_argument("--"+name.replace("_", "-"), type=type(default), default=argparse.SUPPRESS)
    parsed = parser.parse_args(argv)
    run = parsed.run_dir.resolve()
    given = {key: value for key, value in vars(parsed).items() if key in DEFAULTS or key == "recipe"}
    if parsed.predict_images or parsed.evaluate_only:
        if given or parsed.source_run:
            raise ValueError("Inference uses the stored run configuration")
        args = argparse.Namespace(**json.loads((run / "config.json").read_text()))
        inference(run, args, parsed.input or ROOT / ("photo_check" if parsed.predict_images else "test_videos"), parsed.predict_images)
        return
    if parsed.input:
        raise ValueError("--input requires inference mode")
    lock = FileLock(str(ROOT / "cache/spatial_v2/training.lock"), timeout=0)
    try:
        with lock:
            run.mkdir(parents=True, exist_ok=True)
            log = logger_for(run)
            try:
                args, payload = initialize(run, parsed.source_run, given)
                log.info("PATCH RUN %s | parent=%s | split=%s", run, args.source_run, payload["fingerprint"])
                log.info("Experimental recipe; improved generalization requires external evidence. No network downloads.")
                if parsed.prepare_only:
                    log.info("Prepared. Run the same command without --prepare-only")
                    return
                for name in ("ffmpeg", "ffprobe"):
                    if not shutil.which(name):
                        raise RuntimeError(f"Missing {name}")
                model, state = train(run, payload, args, log)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if state["complete"]:
                    final_test(run, payload, args, log)
            except KeyboardInterrupt:
                log.warning("Interrupted; same command resumes last durable checkpoint, up to save_every batches replayed")
            except Exception:
                log.exception("Patch run failed; checkpoint retained")
                raise
    except Timeout:
        raise SystemExit("Another training process is active. Finish it or stop it with Ctrl+C, then run this command again.")


if __name__ == "__main__":
    main()
