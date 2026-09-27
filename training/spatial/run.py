import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from filelock import FileLock

from .common import ROOT, atomic_json, logger_for
from .data import fixed_selection, prepare

DEFAULTS = dict(epochs=20, videos_per_epoch=1000, frames_per_video=4, val_frames=8,
                val_videos=800, test_videos=800, batch_size=8, val_batch_size=8,
                image_size=384, workers=2, decode_workers=2, seed=42, patience=5,
                warmup_epochs=1, backbone_lr=2e-5, head_lr=1e-4,
                cache_gb=20., download_gb=20., reserve_gb=10., max_decode_failure=.02,
                holdout_source="GenVideo_OpenSora", log_every=10, save_every=50,
                offline=False, init_model=None, allow_cpu=False, no_pretrained=False,
                robustness_videos=40, parent_run=None)
MUTABLE = {"batch_size", "val_batch_size", "workers", "decode_workers", "cache_gb",
           "reserve_gb", "log_every", "save_every", "allow_cpu"}


def parse(argv=None):
    parser = argparse.ArgumentParser(description="Audited EfficientNet V2-S training; automatic preparation and resume.")
    parser.add_argument("--fresh", action="store_true", help="New isolated run; preserves old weights and logs")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Build audited data snapshot, without training")
    parser.add_argument("--evaluate-only", action="store_true", help="Test best weights on your own videos, without training")
    parser.add_argument("--predict-images", action="store_true", help="Classify unlabelled photos on CPU without training")
    parser.add_argument("--input", type=Path, help="Video evaluation folder, or unlabelled photo file/folder")
    for name, default in DEFAULTS.items():
        option = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(option, action="store_true", default=argparse.SUPPRESS)
        else:
            parser.add_argument(option, type=type(default) if default is not None else str, default=argparse.SUPPRESS)
    return parser.parse_args(argv)


def configure(parsed):
    base = ROOT / "cache/spatial_v2"
    base.mkdir(parents=True, exist_ok=True)
    latest = base / "latest.json"
    if parsed.run_dir:
        run = parsed.run_dir.resolve()
        if parsed.fresh and (run / "config.json").exists():
            raise ValueError("--fresh requires a new run directory; existing runs are preserved")
    elif not parsed.fresh and latest.exists():
        run = Path(json.loads(latest.read_text())["run_dir"])
    else:
        run = base / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=True)
    config_path = run / "config.json"
    given = {k: v for k, v in vars(parsed).items() if k in DEFAULTS}
    if config_path.exists():
        config = {**DEFAULTS, **json.loads(config_path.read_text())}
        changed = [k for k, v in given.items() if config[k] != v and k not in MUTABLE]
        if changed:
            raise ValueError(f"Cannot change frozen run settings {changed}; use --fresh for a new experiment")
        config.update(given)
    else:
        config = {**DEFAULTS, **given}
    for name in ("epochs", "videos_per_epoch", "frames_per_video", "val_frames", "val_videos", "test_videos",
                 "batch_size", "val_batch_size", "image_size", "decode_workers", "patience", "log_every", "save_every"):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive")
    if config["videos_per_epoch"] < 2 or config["val_videos"] < 2 or config["test_videos"] < 2:
        raise ValueError("Video selections must allow both classes")
    if config["workers"] < 0 or config["warmup_epochs"] < 0 or config["robustness_videos"] < 0:
        raise ValueError("Negative counts are not allowed")
    if not 0 <= config["max_decode_failure"] < .5:
        raise ValueError("max-decode-failure must be in [0, .5)")
    if any(config[k] <= 0 for k in ("cache_gb", "download_gb", "reserve_gb", "backbone_lr", "head_lr")):
        raise ValueError("Budgets and learning rates must be positive")
    if config["init_model"]:
        config["init_model"] = str(Path(config["init_model"]).resolve())
    if config["parent_run"]:
        parent = Path(config["parent_run"]).resolve()
        if parent == run:
            raise ValueError("Fine-tuning requires a new run directory")
        expected = str(parent / "best.pt")
        if config["init_model"] and config["init_model"] != expected:
            raise ValueError("--parent-run must initialize from that parent's best.pt")
        if not Path(expected).is_file():
            raise ValueError(f"Missing parent model: {expected}")
        config.update(parent_run=str(parent), init_model=expected)
    return run, latest, config


def final_evaluation(run, payload, args, logger):
    import torch
    from .engine import build_model, evaluate, write_predictions
    from .media import FrameCache, prepare_frames
    from .metrics import grouped_report
    if (run / "final_report.json").exists():
        logger.info("Final report already exists; closed test not rerun")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(False).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    best = json.loads((run / "best.json").read_text())
    cache = FrameCache(run, args.cache_gb, args.reserve_gb)
    reports = dict(best=best, historical_training_exposure="unknown" if args.init_model else "ImageNet initialization",
                   caveat="In-domain group holdout; exact hashes/known origins checked, near-duplicates not exhaustively ruled out")
    if getattr(args, "parent_run", None):
        lineage = json.loads((run / "lineage.json").read_text())
        reports["historical_training_exposure"] = dict(parent_run=lineage["parent_run"],
            parent_epoch=lineage["parent_epoch"], inherited_train_samples=len(lineage["seen_ids"]),
            note="Parent splits preserved. Old test results were inspected previously, not a fresh blind benchmark.")
    for split in ("test", "ood"):
        records = fixed_selection(payload["splits"][split], args.test_videos, args.seed)
        if not records:
            reports[split] = {"status": "no_data"}
            continue
        prepared, failures = prepare_frames(records, cache, split, args.val_frames, args.seed,
                                             logger, run, args.decode_workers)
        if failures:
            reports[split] = {"status": "incomplete", "requested": len(records), "failures": failures}
            logger.error("%s incomplete; no partial test accuracy published", split)
            continue
        rows, report = evaluate(model, prepared, args, device, logger, split.upper())
        write_predictions(run / f"{split}_predictions.csv", rows)
        report["calibrated_on_validation"] = grouped_report(rows, best["threshold"])
        reports[split] = report
    if args.robustness_videos:
        reports["robustness"] = compression_check(model, payload, run, args, device, cache, logger)
    if (ROOT / "test_videos/real").exists() and (ROOT / "test_videos/fake").exists():
        from .evaluate import external_evaluation
        reports["external"] = external_evaluation(model, ROOT / "test_videos", run, args, device, logger,
            {r["id"] for r in payload["splits"]["train"]}, best["threshold"])
    atomic_json(run / "final_report.json", reports)
    logger.info("FINAL REPORT %s", run / "final_report.json")


def compression_check(model, payload, run, args, device, cache, logger):
    from .engine import evaluate, write_predictions
    from .media import prepare_frames, probe
    from .metrics import grouped_report
    video_pool = [r for r in payload["splits"]["val"] if r.get("kind") != "image_sequence"]
    excluded = len(payload["splits"]["val"]) - len(video_pool)
    logger.info("Compression check: original video containers only; excluded frame-only sequences=%d", excluded)
    records = fixed_selection(video_pool, args.robustness_videos, args.seed)
    original, failures = prepare_frames(records, cache, "robust_original", args.val_frames,
                                         args.seed, logger, run, args.decode_workers)
    if failures:
        return dict(status="incomplete", failures=failures)
    original_rows, _ = evaluate(model, original, args, device, logger, "ROBUST ORIGINAL")
    compressed_rows, errors = [], []
    directory = run / "compression_temp"
    directory.mkdir(exist_ok=True)
    for record in records:
        output = directory / (record["id"] + ".mp4")
        try:
            if shutil.disk_usage(directory).free < (args.reserve_gb * 1024**3 + 256*1024**2):
                raise OSError("Insufficient space for compression check")
            meta = next(r["metadata"] for r in original if r["id"] == record["id"])
            if meta["duration"] > 120:
                raise ValueError("Compression stress-test limit is 120 seconds; original was not truncated")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
                            "-i", record["path"], "-map", "0:v:0", "-an", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "32",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-fs", str(256*1024**2), str(output)],
                           check=True, capture_output=True, timeout=180)
            if abs(probe(output)["duration"] - meta["duration"]) > .5:
                raise ValueError("Compressed output incomplete")
            stat = output.stat()
            transformed = dict(record, path=str(output), size=stat.st_size, mtime=stat.st_mtime_ns)
            prepared, failed = prepare_frames([transformed], cache, "robust_crf32", args.val_frames,
                                               args.seed, logger, run, 1)
            if failed:
                raise RuntimeError(str(failed))
            rows, _ = evaluate(model, prepared, args, device, logger, "ROBUST CRF32")
            compressed_rows.extend(rows)
        except Exception as exc:
            errors.append(dict(id=record["id"], error=str(exc)))
            logger.warning("Compression test error: %s: %s", record["path"], exc)
        finally:
            output.unlink(missing_ok=True)
    paired = {r["id"] for r in compressed_rows}
    originals = [r for r in original_rows if r["id"] in paired]
    write_predictions(run / "compression_original.csv", originals)
    write_predictions(run / "compression_crf32.csv", compressed_rows)
    if not compressed_rows:
        return dict(status="failed", errors=errors)
    by_id = {r["id"]: r for r in originals}
    return dict(status="complete" if not errors else "partial", errors=errors,
                excluded_frame_sequences=excluded,
                original=grouped_report(originals), compressed=grouped_report(compressed_rows),
                prediction_flips=sum((r["score"] >= .5) != (by_id[r["id"]]["score"] >= .5) for r in compressed_rows))


def main(argv=None):
    parsed = parse(argv)
    if parsed.predict_images:
        if parsed.fresh or parsed.prepare_only or parsed.evaluate_only or any(k in DEFAULTS for k in vars(parsed)):
            raise ValueError("--predict-images accepts only --input and --run-dir")
        from .predict_images import main as predict_main
        options = ["--input", str(parsed.input or ROOT / "photo_check")]
        if parsed.run_dir:
            options += ["--run-dir", str(parsed.run_dir)]
        predict_main(options)
        return
    if parsed.evaluate_only:
        if parsed.fresh or parsed.prepare_only or any(k in DEFAULTS for k in vars(parsed)):
            raise ValueError("--evaluate-only accepts only --input and --run-dir; training settings come from the saved run")
        from .evaluate import main as evaluate_main
        options = ["--input", str(parsed.input or ROOT / "test_videos")]
        if parsed.run_dir:
            options += ["--run-dir", str(parsed.run_dir)]
        (ROOT / "cache/spatial_v2").mkdir(parents=True, exist_ok=True)
        with FileLock(str(ROOT / "cache/spatial_v2/training.lock"), timeout=0):
            evaluate_main(options)
        return
    if parsed.input:
        raise ValueError("--input requires --evaluate-only or --predict-images")
    run, latest, config = configure(parsed)
    with FileLock(str(ROOT / "cache/spatial_v2/training.lock"), timeout=0):
        atomic_json(run / "config.json", config)
        atomic_json(latest, dict(run_dir=str(run)))
        logger = logger_for(run)
        logger.info("RUN %s", run)
        logger.info("CONFIG %s", json.dumps(config))
        args = argparse.Namespace(**config)
        try:
            for binary in ("ffmpeg", "ffprobe"):
                if not shutil.which(binary):
                    raise RuntimeError(f"{binary} is missing from PATH")
            if shutil.disk_usage(run).free < args.reserve_gb * 1024**3:
                raise RuntimeError("Free space below reserve before preparation")
            payload = prepare(run, args, logger)
            if parsed.prepare_only:
                logger.info("Preparation complete. Run the same command without --prepare-only to train")
                return
            from .engine import train
            model, _ = train(run, payload, args, logger)
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            final_evaluation(run, payload, args, logger)
        except KeyboardInterrupt:
            logger.warning("Interrupted. Restart the SAME command to restore the last durable checkpoint (up to %d steps replayed).", args.save_every)
        except Exception:
            logger.exception("Run failed; checkpoint preserved. Fix the reported problem and restart the same command")
            raise


if __name__ == "__main__":
    main()
