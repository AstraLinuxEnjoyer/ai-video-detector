"""Train, compare and run the independent spectral/noise branch on CPU."""

import argparse
import json
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import as_completed
from pathlib import Path

import cv2
import numpy as np
from filelock import FileLock
from tqdm import tqdm

from training.data.dfdc import INDEX, default_root, write_index
from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.spatial.data import assert_disjoint
from training.spatial.media import decoding_pool
from training.spatial.metrics import grouped_report, select_threshold
from training.temporal_v2.dashboard import Monitor
from .cohort import build_cohort
from .faces import SHA256 as YUNET_SHA, ensure_weights
from .features import FACE_RECIPE, PATCH_RECIPE, RECIPE, SourceChangedError, extract, preview_frame, verify_record
from .model import VIEWS, balanced_weights, fit, scores_for

DEFAULT_RUN = ROOT / "cache/spectral/runs/spectral_full_frame_v2"
PATCH_RUN = ROOT / "cache/spectral/runs/spectral_residual_v1"
FACE_RUN = ROOT / "cache/spectral/runs/spectral_face_v3"
DEFAULT_PARENT = ROOT / "cache/artifact_spatial/runs/hybrid_v2/splits.json"


def versions():
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True, timeout=15)
    return dict(ffmpeg=ffmpeg.stdout.splitlines()[0], numpy=np.__version__, opencv=cv2.__version__)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_dfdc(root):
    index = write_index(root)
    counts = Counter(f"{r['partition']}/{r['label']}" for r in index["records"])
    return index, dict(parts=len(index["metadata"]), videos=len(index["records"]), counts=dict(counts))


def recipe_for(args):
    return FACE_RECIPE if args.faces else PATCH_RECIPE if args.patches else RECIPE


def prepare(run, args, log, monitor):
    config = dict(recipe=recipe_for(args),
                  versions=versions(), frames=args.frames, seed=args.seed,
                  dfdc_fraction=args.dfdc_fraction, dfdc_root=str(args.dfdc_root.resolve()),
                  parent_splits=str(args.parent_splits.resolve()),
                  counts={split: getattr(args, split) for split in ("train", "val", "test", "ood")})
    if args.faces:
        config.update(yunet_sha256=YUNET_SHA, training_weights="equal_class_source_origin")
    path = run / "spectral_run.json"
    if path.exists():
        saved = load(path)
        cohort = saved["cohort"]
        if saved["config"] != config or fingerprint(cohort["splits"]) != cohort["fingerprint"]:
            raise ValueError("Frozen spectral recipe/cohort changed; use the original options or a new run directory")
        assert_disjoint(cohort["splits"])
    else:
        index, summary = index_dfdc(args.dfdc_root)
        log.info("DFDC index: %s | %s", INDEX, summary)
        cohort = build_cohort(args.parent_splits, index, config["counts"], args.dfdc_fraction, args.seed,
                              hash_database=ROOT/"cache/datasets/dfdc_hashes.sqlite" if args.faces else None,
                              publish=monitor.publish if args.faces else None)
        atomic_json(path, dict(config=config, cohort=cohort))
    log.info("All train=%s | duplicate-family exclusions=%d known-label exclusions=%d",
             args.train == 0, len(cohort.get("dfdc_duplicate_exclusions", [])), len(cohort.get("known_label_exclusions", [])))
    for split, rows in cohort["splits"].items():
        log.info("Cohort %s | %s", split, dict(Counter(f"{r['source']}/{r['label']}" for r in rows)))
    return config, cohort


def publish_preview(monitor, record, recipe, log, split, models=None):
    try:
        verify_record(record)
        preview = preview_frame(record, recipe)
        verify_record(record)
        preview["split"] = split
        if models is not None:
            mode = "full_face" if "full_face" in models else "combined"
            value = float(scores_for(models, mode, [record])[0])
            threshold = models[mode]["threshold"]
            preview.update(score=value, threshold=threshold, model=mode,
                           prediction="FAKE" if value >= threshold else "REAL")
            if "face" in models and "clean" in record.get("face_vectors", {}):
                preview["face_score"] = float(scores_for(models, "face", [record])[0])
            if mode == "full_face":
                preview["full_score"] = float(scores_for(models, "combined", [record])[0])
        monitor.publish(preview=preview, spectrum=preview["spectrum"])
    except (ValueError, OSError, SourceChangedError, subprocess.SubprocessError, cv2.error) as exc:
        log.warning("Dashboard preview skipped: %s", exc)


def feature_jobs(pool, records, workers, load_one):
    # Keep only a small queue of decodes, even when training on the entire catalog.
    size = 4*workers
    for start in range(0, len(records), size):
        pending = {pool.submit(load_one, r): r for r in records[start:start+size]}
        for future in as_completed(pending):
            yield future, pending[future]


def extract_split(run, split, records, config, workers, log, monitor, models=None, face_weights=None):
    directory = run / "features" / split
    signature = fingerprint([config["recipe"], config["versions"], config["frames"], split])

    def one(record):
        verify_record(record)
        path = directory / f"{record['id']}.json"
        key = fingerprint([signature, record["id"], record["size"], record["mtime"]])
        if path.exists():
            saved = load(path)
            if saved["signature"] == key:
                return dict(record, **saved["features"]), True
        features = extract(record["path"], config["frames"], stress=split != "train", recipe=config["recipe"],
                           face_weights=face_weights)
        verify_record(record)
        atomic_json(path, dict(signature=key, features=features))
        return dict(record, **features), False

    usable, failures = [], []
    cached = 0
    faces = 0
    last = time.monotonic()
    preview_id = None
    monitor.publish(phase=split, completed=0, total=len(records), usable=0, failed=0, cached=0,
                    face_videos=0 if face_weights else None, detail=None)
    with decoding_pool(workers) as pool:
        jobs = feature_jobs(pool, records, workers, one)
        for completed, (future, record) in enumerate(tqdm(jobs, total=len(records), desc=f"SPECTRAL {split}"), 1):
            try:
                row, hit = future.result()
                usable.append(row)
                cached += int(hit)
                faces += int("clean" in row.get("face_vectors", {}))
            except SourceChangedError:
                raise
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                failure = {k: record[k] for k in ("id", "group", "source", "label", "path")}
                failures.append(dict(failure, error=str(exc)))
                log.warning("No spectral score [%s] %s: %s", split, record["path"], exc)
            if completed == len(records) or time.monotonic() - last >= (4 if len(records) > 1000 else 1):
                monitor.publish(completed=completed, usable=len(usable), failed=len(failures), cached=cached,
                                face_videos=faces if face_weights else None)
                if usable and usable[-1]["id"] != preview_id:
                    publish_preview(monitor, usable[-1], config["recipe"], log, split, models)
                    preview_id = usable[-1]["id"]
                last = time.monotonic()
            if completed % 1000 == 0:
                log.info("Features %s %d/%d | usable=%d with_face=%d cached=%d",
                         split, completed, len(records), len(usable), faces, cached)
    usable.sort(key=lambda row: row["id"])
    failures.sort(key=lambda row: row["id"])
    coverage = dict(requested=len(records), usable=len(usable), failures=failures,
                    requested_by_class_source=dict(Counter(f"{r['label']}/{r['source']}" for r in records)),
                    usable_by_class_source=dict(Counter(f"{r['label']}/{r['source']}" for r in usable)),
                    faces_by_treatment={t: dict(Counter(f"{r['label']}/{r['source']}" for r in usable
                                       if t in r.get("face_vectors", {}))) for t in ("clean", "jpeg70")})
    log.info("Coverage %s requested=%d usable=%d failures=%d", split, len(records), len(usable), len(failures))
    atomic_json(run / f"{split}_coverage.json", coverage)
    return usable, cached


def report_pairs(rows, requested):
    groups = defaultdict(dict)
    for row in rows:
        if row["source"] == "DFDC":
            groups[row["group"]][row["label"]] = row["score"]
    paired = [pair for pair in groups.values() if set(pair) == {0, 1}]
    wins = sum(pair[1] > pair[0] for pair in paired)
    ties = sum(pair[1] == pair[0] for pair in paired)
    return dict(requested=len({r["group"] for r in requested if r["source"] == "DFDC"}),
                complete=len(paired), fake_ranked_higher=wins, ties=ties,
                pair_accuracy=(wins + .5*ties)/len(paired) if paired else None)


def experiment(run, config, cohort, workers, log, monitor, face_weights=None):
    import torch

    torch.set_num_threads(min(torch.get_num_threads(), 2))
    if face_weights:
        cv2.setNumThreads(1)
    data, cached_counts = {}, {}

    def collect(split, fitted=None):
        data[split], cached_counts[split] = extract_split(run, split, cohort["splits"][split],
                                                        config, workers, log, monitor, fitted, face_weights)
        if split in ("train", "val", "test"):
            for dfdc in (False, True):
                labels = {r["label"] for r in data[split] if (r["source"] == "DFDC") == dfdc}
                if labels != {0, 1}:
                    raise ValueError(f"{split} needs usable videos of both classes in DFDC and old sources")
    reports = dict(cohort_fingerprint=cohort["fingerprint"], variants={},
                   caveats=["Not a trained ensemble or a pixel localization benchmark.",
                            "Existing project splits were inspected before; OOD may be fake-only.",
                            "JPEG70 is a frame-level stress proxy, not video transcoding.",
                            "Nominal 5% validation FPR is not a guarantee on new sources."])
    models = {}
    predictions = []
    collect("train")
    train = data["train"]
    training_vectors = [r["vectors"]["clean"] for r in train]
    train_labels = [r["label"] for r in train]
    weights = balanced_weights(train) if face_weights else None
    if weights is not None:
        masses = defaultdict(float)
        for row, weight in zip(train, weights):
            masses[f"{row['label']}/{row['source']}"] += float(weight)
        reports["training_weight_mass"] = dict(masses)
    face_train = [r for r in train if "clean" in r.get("face_vectors", {})]
    use_face = bool(face_weights) and {r["label"] for r in face_train} == {0, 1}
    monitor.publish(phase="fit", completed=0, total=len(VIEWS)+int(use_face))
    for completed, mode in enumerate(VIEWS, 1):
        monitor.publish(active_variant=mode)
        model = fit(training_vectors, train_labels, mode, weights)
        models[mode] = model
        log.info("Fit %s | train=%d features=%d", mode, len(train), len(model["weights"]))
        monitor.publish(completed=completed)
    if use_face:
        monitor.publish(active_variant="face")
        models["face"] = fit([r["face_vectors"]["clean"] for r in face_train],
                             [r["label"] for r in face_train], "combined", balanced_weights(face_train))
        models["full_face"] = dict(face_weight=.5)
        monitor.publish(completed=len(VIEWS)+1)
        log.info("Fit face | train=%d/%d; full + face uses an equal blend when a face is found",
                 len(face_train), len(train))
    elif face_weights:
        log.warning("Face branch unavailable: detected faces must include both training classes")
    collect("val")
    val = data["val"]
    if use_face and {r["label"] for r in val if "clean" in r.get("face_vectors", {})} != {0, 1}:
        log.warning("Face branch unavailable: need both classes with detected faces on validation")
        del models["face"], models["full_face"]
        use_face = False
    if face_weights:
        reports["face_status"] = "available" if use_face else "insufficient_classes"
    for mode, model in models.items():
        scores = scores_for(models, mode, val)
        mask = np.isfinite(scores)
        model["threshold"] = select_threshold(np.asarray([r["label"] for r in val])[mask], scores[mask])
        log.info("Validation threshold %s | %.6f", mode, model["threshold"])
    publish_preview(monitor, val[-1], config["recipe"], log, "val", models)
    collect("test", models)
    collect("ood", models)
    completed = 0
    monitor.publish(phase="evaluate", completed=completed,
                    total=2*len(models)*sum(bool(data[s]) for s in ("val", "test", "ood")))
    for mode, model in models.items():
        threshold = model["threshold"]
        mode_report = {}
        for split in ("val", "test", "ood"):
            if not data[split]:
                continue
            for treatment in ("clean", "jpeg70"):
                samples = data[split]
                scores = scores_for(models, mode, samples, treatment)
                rows = [dict({k: r[k] for k in ("id", "source", "label", "group")},
                             score=float(p), model=mode, split=split, treatment=treatment,
                             has_face=treatment in r.get("face_vectors", {}))
                        for r, p in zip(samples, scores) if np.isfinite(p)]
                predictions.extend(rows)
                report = grouped_report(rows, threshold) if rows else dict(
                    n=0, balanced_accuracy=None, false_positive_rate=None, recall_fake=None)
                report["requested"] = len(cohort["splits"][split])
                report["coverage"] = len(rows) / report["requested"]
                report["dfdc_pairs"] = report_pairs(rows, cohort["splits"][split])
                matched = [r for r in rows if r["has_face"]]
                if face_weights:
                    report["face_subset"] = grouped_report(matched, threshold) if matched else None
                mode_report[f"{split}/{treatment}"] = report
                log.info("%s %s/%s | n=%d/%d balanced=%s FPR=%s recall_fake=%s DFDC=%s",
                         mode, split, treatment, len(rows), report["requested"], report["balanced_accuracy"],
                         report["false_positive_rate"], report["recall_fake"], report["dfdc_pairs"])
                completed += 1
                monitor.publish(completed=completed, active_variant=mode, evaluation=f"{split}/{treatment}")
        reports["variants"][mode] = mode_report
    atomic_json(run / "predictions.json", predictions)
    atomic_json(run / "final_report.json", reports)
    atomic_json(run / "model.json", dict(config=config, models=models, cohort_fingerprint=cohort["fingerprint"]))
    log.info("COMPLETE | %s", run / "final_report.json")
    usable = sum(map(len, data.values()))
    requested = sum(map(len, cohort["splits"].values()))
    candidates = [r for r in data["test"] if r["source"] == "DFDC" and r["label"] == 1]
    sample = next((r for r in candidates if "clean" in r.get("face_vectors", {})),
                  candidates[0] if candidates else data["test"][-1])
    publish_preview(monitor, sample, config["recipe"], log, "test", models)
    monitor.publish(status="complete", phase="complete", finished=time.time(),
                    completed=requested, total=requested, usable=usable,
                    failed=requested-usable, cached=sum(cached_counts.values()),
                    face_videos=sum("clean" in r.get("face_vectors", {}) for rows in data.values() for r in rows)
                    if face_weights else None)


def predict(run, path):
    saved = load(run / "model.json")
    config = saved["config"]
    if config["recipe"] not in (RECIPE, PATCH_RECIPE, FACE_RECIPE) or config["versions"] != versions():
        raise ValueError("Model requires the spectral recipe and decoder/library versions used in training")
    face_weights = None
    if config["recipe"] == FACE_RECIPE:
        if config.get("yunet_sha256") != YUNET_SHA:
            raise ValueError("Face detector does not match the trained model")
        face_weights = ensure_weights()
    features = extract(path, config["frames"], recipe=config["recipe"], face_weights=face_weights)
    models = saved["models"]
    mode = "full_face" if "full_face" in models else "combined"
    value = float(scores_for(models, mode, [features])[0])
    threshold = models[mode]["threshold"]
    print(json.dumps(dict(video=str(path.resolve()), model=config["recipe"], score=value,
                          variant=mode, threshold=threshold, suspicious=value >= threshold,
                          timestamps=features["timestamps"], patches=features["patches"],
                          face_frames=features.get("face_counts", {}).get("clean")), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    recipes = parser.add_mutually_exclusive_group()
    recipes.add_argument("--patches", action="store_true", help="Use the legacy 128px patch experiment")
    recipes.add_argument("--faces", action="store_true", help="Whole-frame and face analysis; all training videos by default")
    parser.add_argument("--parent-splits", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--dfdc-root", type=Path, default=default_root())
    parser.add_argument("--dfdc-fraction", type=float, default=.25)
    parser.add_argument("--train", type=int, help="Bound training videos; 0 uses all train with --faces")
    parser.add_argument("--val", type=int, default=160)
    parser.add_argument("--test", type=int, default=240)
    parser.add_argument("--ood", type=int, default=120)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8767)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--index-only", action="store_true")
    action.add_argument("--predict", type=Path)
    action.add_argument("--dashboard", action="store_true")
    action.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir or (FACE_RUN if args.faces else PATCH_RUN if args.patches else DEFAULT_RUN)
    if args.train is None:
        args.train = 0 if args.faces else 400
    if args.dashboard:
        from .dashboard import serve

        serve(args.run_dir, args.port)
        return
    if args.index_only:
        _, summary = index_dfdc(args.dfdc_root)
        print(json.dumps(dict(index=str(INDEX), **summary), indent=2))
        return
    run = args.run_dir.resolve()
    if args.predict:
        predict(run, args.predict)
        return
    if (not 0 < args.dfdc_fraction <= .75 or not 1 <= args.frames <= 12 or not 1 <= args.workers <= 8
            or (not (args.faces and args.train == 0) and (not 4 <= args.train <= 10000 or args.train % 2))
            or any(not 4 <= n <= 10000 or n % 2 for n in (args.val, args.test))
            or not 0 <= args.ood <= 10000):
        parser.error("Invalid cohort quota/counts, frames or workers; train/val/test counts must be even")
    if (run / "splits.json").exists() or (run / "checkpoint.pt").exists():
        parser.error("Use a separate spectral run directory")
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(run / ".spectral.lock", timeout=0):
        log = logger_for(run)
        monitor = Monitor(run)
        try:
            monitor.publish(recipe=recipe_for(args),
                            status="running", phase="prepare", started=time.time(),
                            completed=0, total=None, usable=0, failed=0, cached=0,
                            active_variant=None, evaluation=None, preview=None, spectrum=None, plan={}, dfdc={},
                            face_videos=None, detail=None)
            config, cohort = prepare(run, args, log, monitor)
            monitor.publish(plan={split: len(rows) for split, rows in cohort["splits"].items()},
                            dfdc={split: sum(r["source"] == "DFDC" for r in rows)
                                  for split, rows in cohort["splits"].items()})
            if args.prepare_only:
                monitor.publish(status="prepared", completed=None, total=None, detail=None, finished=time.time())
                log.info("PREPARED | rerun without --prepare-only to extract features and train")
                return
            weights = ensure_weights() if args.faces else None
            experiment(run, config, cohort, args.workers, log, monitor, weights)
        except KeyboardInterrupt:
            monitor.publish(status="stopped", finished=time.time())
            log.info("Interrupted; repeat the same command to reuse prepared features")
            raise
        except Exception:
            monitor.publish(status="error", finished=time.time())
            log.exception("Spectral run failed; existing weights and source videos were not modified")
            raise
        finally:
            monitor.close()


if __name__ == "__main__":
    main()
