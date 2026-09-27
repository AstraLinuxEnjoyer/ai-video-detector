import csv
import io
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from tqdm import tqdm

from .common import SCHEMA, atomic_json, event
from .data import Rotation, fixed_selection
from .media import FrameCache, prepare_frames
from .metrics import bootstrap_interval, grouped_report, metrics, select_threshold
from .lineage import file_sha


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


class Frames(Dataset):
    def __init__(self, records, size, train=False, seed=42):
        self.records = records
        self.samples = [(r, i, p) for r in records for i, p in enumerate(r["frames"])]
        self.size, self.train, self.seed = size, train, seed
        if train:
            random.Random(seed).shuffle(self.samples)
        self.crop = transforms.RandomResizedCrop(size, scale=(.65, 1), ratio=(.75, 1.3333))
        self.eval_crop = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size)])
        self.normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            [.485, .456, .406], [.229, .224, .225])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record, frame_index, path = self.samples[index]
        with Image.open(path) as original:
            image = original.convert("RGB")
        rng = random.Random(f"{self.seed}:{record['id']}:{frame_index}")
        if self.train:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(rng.randrange(2**31))
                image = self.crop(image)
                image = transforms.ColorJitter(.1, .1, .1)(image)
            if rng.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < .3:
                small = rng.randint(max(32, self.size // 2), self.size)
                image = image.resize((small, small), Image.Resampling.BILINEAR).resize(
                    (self.size, self.size), Image.Resampling.BILINEAR)
            if rng.random() < .3:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=rng.randint(45, 95))
                buffer.seek(0)
                with Image.open(buffer) as decoded:
                    image = decoded.convert("RGB")
            if rng.random() < .1:
                image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.1, .6)))
        else:
            image = self.eval_crop(image)
        return self.normalize(image), float(record["label"]), record["id"], frame_index


def init_worker(_worker):
    torch.set_num_threads(1)


def loader(dataset, batch_size, workers, offset=0):
    if offset:
        dataset = Subset(dataset, range(offset, len(dataset)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=False,
                      worker_init_fn=init_worker, generator=torch.Generator().manual_seed(42),
                      **({"prefetch_factor": 2} if workers else {}))


def build_model(pretrained=True):
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    return model


def atomic_torch(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def save_state(run, model, optimizer, scheduler, scaler, state):
    if shutil.disk_usage(run).free < (state.get("reserve_gb", 0) + .75) * 1024**3:
        raise OSError("Not enough space to atomically save checkpoint while preserving the disk reserve")
    payload = dict(state, model=model.state_dict(), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                   torch_rng=torch.get_rng_state(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    atomic_torch(run / "checkpoint.pt", payload)


def write_predictions(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        fields = ["id", "path", "group", "source", "label", "score", "max_score", "frames"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model, records, args, device, logger, tag):
    model.eval()
    dataset = Frames(records, args.image_size)
    scores = defaultdict(list)
    frame_y, frame_p = [], []
    started = time.monotonic()
    with torch.inference_mode():
        for index, (images, labels, ids, _) in enumerate(tqdm(
                loader(dataset, args.val_batch_size, args.workers), desc=tag)):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(images.to(device, non_blocking=True)).flatten()
            probabilities = output.float().sigmoid().cpu().tolist()
            if not all(math.isfinite(p) for p in probabilities):
                raise RuntimeError("Nonfinite validation predictions")
            frame_y.extend(labels.tolist())
            frame_p.extend(probabilities)
            for item, p in zip(ids, probabilities):
                scores[item].append(p)
            if (index + 1) % 50 == 0:
                logger.info("%s batches=%d frames=%d", tag, index + 1, len(frame_y))
    rows = [dict(id=r["id"], path=r["path"], source=r["source"], group=r["group"], label=r["label"],
                 score=float(np.mean(scores[r["id"]])), max_score=max(scores[r["id"]]),
                 frames=len(scores[r["id"]])) for r in records]
    if any(r["frames"] != len(original["frames"]) for r, original in zip(rows, records)):
        raise RuntimeError("Incomplete validation coverage")
    report = grouped_report(rows)
    report["frame_metrics"] = metrics(frame_y, frame_p)
    report["elapsed_seconds"] = time.monotonic() - started
    report["balanced_accuracy_ci95"] = bootstrap_interval(rows, args.seed)
    logger.info("%s videos=%d accuracy=%.4f balanced_acc=%s AUROC=%s FPR=%s recall_fake=%s CI95=%s",
                tag, report["n"], report["accuracy"], report["balanced_accuracy"], report["auroc"],
                report["false_positive_rate"], report["recall_fake"], report["balanced_accuracy_ci95"])
    for source, values in report["by_source"].items():
        logger.info("%s SOURCE %s n=%d accuracy=%.4f FP=%d FN=%d score_quantiles=%s", tag, source,
                    values["n"], values["accuracy"], values["fp"], values["fn"], values["score_quantiles"])
    return rows, report


def metadata_baseline(train_records, val_records):
    # JPEG sequences have no trustworthy source FPS, duration or container bitrate.
    excluded = dict(train=sum(r.get("kind") == "image_sequence" for r in train_records),
                    val=sum(r.get("kind") == "image_sequence" for r in val_records))
    train_records = [r for r in train_records if r.get("kind") != "image_sequence"]
    val_records = [r for r in val_records if r.get("kind") != "image_sequence"]
    if not train_records or not val_records:
        return dict(status="no_video_metadata", excluded_frame_sequences=excluded)
    def features(records):
        return np.asarray([[math.log1p(r["metadata"]["width"]), math.log1p(r["metadata"]["height"]),
                            math.log1p(r["metadata"]["fps"]), math.log1p(r["metadata"]["duration"]),
                            math.log1p(r["size"] / r["metadata"]["duration"])] for r in records])
    x, v = features(train_records), features(val_records)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
    x = np.column_stack([np.ones(len(x)), (x - mean) / std])
    v = np.column_stack([np.ones(len(v)), (v - mean) / std])
    beta = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ np.asarray([r["label"] for r in train_records]))
    scores = np.clip(v @ beta, 0, 1)
    return dict(metrics([r["label"] for r in val_records], scores), excluded_frame_sequences=excluded)


def optimizer_batch(model, images, labels, optimizer, scaler, logger, run, context):
    """Retry numerical AMP failures without losing samples or updating BN twice."""
    use_amp = images.device.type == "cuda" and scaler.is_enabled()
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(images.device) if images.is_cuda else None

    def restore():
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, images.device)

    for attempt in range(3 if use_amp else 1):
        amp_attempt = use_amp and attempt < 2
        if attempt:
            restore()
        optimizer.zero_grad(set_to_none=True)
        scale_before = scaler.get_scale()
        with torch.amp.autocast("cuda", enabled=amp_attempt):
            logits = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        reason = None
        if not torch.isfinite(loss):
            reason = "nonfinite_loss"
        else:
            if amp_attempt:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
            if not finite:
                reason = "nonfinite_gradients"
                if amp_attempt:
                    # unscale_ recorded the nonfinite gradients: step MUST skip the update.
                    scaler.step(optimizer)
                    scaler.update()
            else:
                try:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                except RuntimeError:
                    restore()
                    raise
                if amp_attempt:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if attempt:
                    detail = dict(context, event="batch_recovered", attempts=attempt+1,
                                  precision="fp16" if amp_attempt else "fp32", scale=scaler.get_scale())
                    logger.warning("NUMERICS %s", json.dumps(detail))
                    event(run / "numerics.jsonl", detail)
                return loss.detach(), logits.detach(), norm, attempt
        detail = dict(context, event="retry" if amp_attempt else "fatal", reason=reason,
                      attempt=attempt+1, scale_before=scale_before, scale_after=scaler.get_scale())
        logger.warning("NUMERICS %s", json.dumps(detail))
        event(run / "numerics.jsonl", detail)
        if not amp_attempt:
            restore()
            raise RuntimeError("Nonfinite loss/gradients persist in FP32; no optimizer update or sample advancement. See numerics.jsonl")
    raise AssertionError("Unreachable numerical retry state")


def train(run, payload, args, logger):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable. Fix the torch environment or explicitly use --allow-cpu")
    logger.info("torch=%s device=%s gpu=%s AMP=%s", torch.__version__, device,
                torch.cuda.get_device_name() if device.type == "cuda" else "none", device.type == "cuda")
    environment = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                       torchvision=torchvision.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                       cudnn=torch.backends.cudnn.version(),
                       gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                       seed=args.seed, split_hash=payload["fingerprint"])
    atomic_json(run / "environment.json", environment)
    logger.info("ENVIRONMENT %s", json.dumps(environment))
    checkpoint = run / "checkpoint.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.exists() else None
    lineage = json.loads((run / "lineage.json").read_text()) if getattr(args, "parent_run", None) else None
    model = build_model(pretrained=not saved and not args.init_model and not args.no_pretrained).to(device)
    if args.init_model and not saved:
        if lineage and file_sha(args.init_model) != lineage["weights_sha256"]:
            raise RuntimeError("Parent weights changed after preparation")
        weights = torch.load(args.init_model, map_location="cpu", weights_only=True)
        model.load_state_dict(weights.get("model", weights))
        logger.info("Warm start from %s; history=%s", args.init_model,
                    "audited parent; old splits preserved" if lineage else "UNKNOWN")
    optimizer = torch.optim.AdamW([
        dict(params=model.features.parameters(), lr=args.backbone_lr),
        dict(params=model.classifier.parameters(), lr=args.head_lr)], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    state = dict(schema=SCHEMA, split_hash=payload["fingerprint"], epoch=0, offset=0, plan=None,
                 rotation=None, seen=[], quarantine=[], best=-1., stale=0, progress=None, complete=False,
                 reserve_gb=args.reserve_gb)
    if saved:
        if saved["split_hash"] != payload["fingerprint"] or saved["schema"] != SCHEMA:
            raise RuntimeError("Checkpoint and split fingerprint disagree")
        model.load_state_dict(saved.pop("model"))
        optimizer.load_state_dict(saved.pop("optimizer"))
        scheduler.load_state_dict(saved.pop("scheduler"))
        scaler.load_state_dict(saved.pop("scaler"))
        torch.set_rng_state(saved.pop("torch_rng"))
        random.setstate(saved.pop("python_rng"))
        np.random.set_state(saved.pop("numpy_rng"))
        cuda_rng = saved.pop("cuda_rng")
        if device.type == "cuda" and cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        state = saved
        state["reserve_gb"] = args.reserve_gb
        logger.info("RESUME epoch=%d processed_frames=%d best=%.4f", state["epoch"]+1, state["offset"], state["best"])
    if state["complete"]:
        logger.info("Training already complete. Final evaluation can resume if it was interrupted.")
        return model, state
    splits = payload["splits"]
    inherited = set(lineage["seen_ids"]) if lineage else set()
    rotation = Rotation(splits["train"], args.seed, state["rotation"], previously_seen=inherited)
    logger.info("Sampling: class/source quotas; unseen parent samples first within each source on its first cycle")
    lookup = {r["id"]: r for r in splits["train"]}
    cache = FrameCache(run, args.cache_gb, args.reserve_gb)
    for completed_epoch in range(state["epoch"]):
        cache.clear_train(completed_epoch)
    val_freeze = run / "validation_ids.json"
    val_selection = fixed_selection(splits["val"], args.val_videos, args.seed)
    if val_freeze.exists():
        accepted = set(json.loads(val_freeze.read_text())["accepted"])
        val_selection = [r for r in val_selection if r["id"] in accepted]
    val_records, val_failed = prepare_frames(val_selection, cache, "val", args.val_frames,
                                            args.seed, logger, run, args.decode_workers)
    if val_failed and (val_freeze.exists() or len(val_failed) / len(val_selection) > args.max_decode_failure):
        raise RuntimeError("Validation decoding incomplete; existing frozen validation cannot change. See val_decode.json")
    if not val_freeze.exists():
        atomic_json(val_freeze, dict(requested=len(val_selection), accepted=[r["id"] for r in val_records], excluded=val_failed))
        logger.info("Validation frozen BEFORE optimization: accepted=%d excluded=%d", len(val_records), len(val_failed))
    if {r["label"] for r in val_records} != {0, 1}:
        raise RuntimeError("Validation must retain both labels")
    if lineage and state["best"] < 0:
        rows, initial = evaluate(model, val_records, args, device, logger, "INITIAL PARENT")
        threshold = select_threshold([r["label"] for r in rows], [r["score"] for r in rows])
        atomic_json(run / "initial_val.json", initial)
        write_predictions(run / "initial_val.csv", rows)
        state["best"] = initial["balanced_accuracy"]
        atomic_torch(run / "best.pt", model.state_dict())
        atomic_json(run / "best.json", dict(epoch=0, balanced_accuracy=state["best"],
                    threshold=threshold, split_hash=payload["fingerprint"], image_size=args.image_size,
                    preprocessing="ffmpeg_rgb_hdr_tonemap_resize_short_center_crop_imagenet",
                    frames=args.val_frames, aggregation="mean"))
        save_state(run, model, optimizer, scheduler, scaler, state)
        logger.info("Initial parent baseline saved: %.4f; best weights only replaced on improvement", state["best"])
    for epoch in range(state["epoch"], args.epochs):
        state["epoch"] = epoch
        started = time.monotonic()
        logger.info("EPOCH %d/%d", epoch+1, args.epochs)
        if state["plan"] is None:
            planned = [r for r in rotation.take(args.videos_per_epoch) if r["id"] not in state["quarantine"]]
            state["rotation"] = rotation.state
            state["plan"] = [r["id"] for r in planned]
            state["progress"] = dict(labels=[], scores=[], seen=[], steps=0, loss_sum=0., frames=0)
            # Persist selection before decoding so interruption cannot advance the queues twice.
            save_state(run, model, optimizer, scheduler, scaler, state)
        planned = [lookup[i] for i in state["plan"]]
        records, failures = prepare_frames(planned, cache, f"train_{epoch}", args.frames_per_video,
                                           args.seed + epoch, logger, run, args.decode_workers)
        if failures:
            if state["offset"] or len(failures)/max(len(planned), 1) > args.max_decode_failure:
                raise RuntimeError("Training decode failure limit exceeded or cursor would change; see data_errors.jsonl")
            state["quarantine"] = sorted(set(state["quarantine"]) | {r["id"] for r in failures})
            state["plan"] = [r["id"] for r in records]
            logger.warning("Quarantined %d videos BEFORE optimization; no replacement samples", len(failures))
            save_state(run, model, optimizer, scheduler, scaler, state)
        if {r["label"] for r in records} != {0, 1}:
            raise RuntimeError("Training selection must contain both classes")
        atomic_json(run / f"epoch_{epoch+1:03}_plan.json", dict(ids=state["plan"],
                    sources=dict(Counter(r["source"] for r in records)),
                    labels=dict(Counter(r["label"] for r in records))))
        logger.info("Epoch videos=%d labels=%s sources=%s", len(records),
                    dict(Counter(r["label"] for r in records)), dict(Counter(r["source"] for r in records)))
        if epoch == 0:
            baseline = metadata_baseline(records, val_records)
            atomic_json(run / "metadata_only_baseline.json", baseline)
            logger.warning("METADATA-ONLY baseline (no pixels): %s. High accuracy indicates dataset shortcuts.", baseline)
        model.train()
        frozen = epoch < args.warmup_epochs
        for parameter in model.features.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            model.features.eval()
        logger.info("Backbone frozen=%s LR=%s", frozen, [g["lr"] for g in optimizer.param_groups])
        dataset = Frames(records, args.image_size, train=True, seed=args.seed + epoch)
        progress = state["progress"]
        iterator = loader(dataset, args.batch_size, args.workers, state["offset"])
        previous = time.monotonic()
        loop = tqdm(iterator, desc=f"TRAIN {epoch+1}", initial=state["offset"] // args.batch_size,
                    total=math.ceil(len(dataset) / args.batch_size))
        for images, labels, ids, frame_indices in loop:
            data_seconds = time.monotonic() - previous
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32).unsqueeze(1)
            before = model.classifier[1].weight.detach().clone()
            loss, logits, norm, retries = optimizer_batch(model, images, labels, optimizer, scaler,
                logger, run, dict(epoch=epoch+1, step=progress["steps"]+1, offset=state["offset"],
                                  video_ids=list(ids), frame_indices=frame_indices.tolist()))
            progress["numerical_retries"] = progress.get("numerical_retries", 0) + retries
            delta = float((model.classifier[1].weight.detach() - before).abs().mean())
            if not math.isfinite(delta):
                raise RuntimeError("Nonfinite classifier weight update")
            progress["unchanged_steps"] = progress.get("unchanged_steps", 0) + 1 if delta == 0 else 0
            if delta == 0:
                logger.warning("Classifier weights unchanged on this step; consecutive=%d", progress["unchanged_steps"])
            if progress["unchanged_steps"] >= 10:
                raise RuntimeError("Classifier weights unchanged for 10 consecutive steps; inspect loss, gradients and learning rate")
            count = len(ids)
            progress["loss_sum"] += float(loss.detach()) * count
            progress["frames"] += count
            progress["steps"] += 1
            progress["labels"].extend(labels.detach().cpu().flatten().tolist())
            progress["scores"].extend(logits.detach().float().sigmoid().cpu().flatten().tolist())
            progress["seen"] = sorted(set(progress["seen"]).union(ids))
            state["seen"] = sorted(set(state["seen"]).union(ids))
            state["offset"] += count
            loop.set_postfix(loss=f"{float(loss.detach()):.4f}", samples=state["offset"])
            if progress["steps"] == 1 or progress["steps"] % args.log_every == 0:
                details = dict(epoch=epoch+1, step=progress["steps"], frames=state["offset"],
                               loss=float(loss.detach()), average_loss=progress["loss_sum"]/progress["frames"],
                               grad_norm=float(norm), classifier_update_mean=delta, data_wait_s=data_seconds,
                               amp_scale=scaler.get_scale(), numerical_retries=progress["numerical_retries"],
                               lr=[g["lr"] for g in optimizer.param_groups],
                               gpu_peak_gb=torch.cuda.max_memory_allocated()/1024**3 if device.type == "cuda" else 0)
                logger.info("BATCH %s", json.dumps(details))
                event(run / "batches.jsonl", details)
            if progress["steps"] % args.save_every == 0:
                save_state(run, model, optimizer, scheduler, scaler, state)
            previous = time.monotonic()
        # Validation can be retried without repeating completed optimizer updates.
        save_state(run, model, optimizer, scheduler, scaler, state)
        rows, report = evaluate(model, val_records, args, device, logger, f"VAL {epoch+1}")
        write_predictions(run / f"epoch_{epoch+1:03}_val.csv", rows)
        train_report = metrics(progress["labels"], progress["scores"])
        report["train"] = train_report
        report["train_loss"] = progress["loss_sum"] / progress["frames"]
        report["numerical_retries"] = progress.get("numerical_retries", 0)
        report["epoch"] = epoch+1
        report["elapsed_seconds"] = time.monotonic() - started
        seen = set(state["seen"])
        report["coverage"] = {source: dict(pool=len(pool), seen=sum(r["id"] in seen for r in pool))
                              for source in sorted({r["source"] for r in splits["train"]})
                              for pool in [[r for r in splits["train"] if r["source"] == source]]}
        report["unique_train_seen"] = len(seen)
        report["inherited_train_seen"] = len(inherited)
        report["unique_train_seen_including_parent"] = len(seen | inherited)
        report["train_pool"] = len(lookup)
        report["quarantined_train_ids"] = state["quarantine"]
        report["validation_preflight"] = json.loads(val_freeze.read_text())
        threshold = select_threshold([r["label"] for r in rows], [r["score"] for r in rows])
        report["validation_calibrated_5pct_fpr"] = grouped_report(rows, threshold)
        if report["balanced_accuracy"] >= .98:
            logger.warning("Very high validation accuracy: inspect metadata baseline, held-out sources and robustness; this is not proof of generalization")
        score = report["balanced_accuracy"]
        if score > state["best"] + 1e-4:
            state["best"], state["stale"] = score, 0
            atomic_torch(run / "best.pt", model.state_dict())
            atomic_json(run / "best.json", dict(epoch=epoch+1, balanced_accuracy=score,
                        threshold=threshold, split_hash=payload["fingerprint"], image_size=args.image_size,
                        preprocessing="ffmpeg_rgb_hdr_tonemap_resize_short_center_crop_imagenet",
                        frames=args.val_frames, aggregation="mean"))
            logger.info("BEST updated: balanced_accuracy=%.4f", score)
        else:
            state["stale"] += 1
        scheduler.step(score)
        atomic_torch(run / "last.pt", model.state_dict())
        atomic_json(run / f"epoch_{epoch+1:03}_metrics.json", report)
        event(run / "epochs.jsonl", report)
        atomic_json(run / "coverage.json", dict(seen_ids=state["seen"], by_source=report["coverage"]))
        logger.info("EPOCH RESULT train_acc=%.4f val_bal_acc=%.4f AUC=%.4f FPR=%.4f recall_fake=%.4f seen=%d/%d stale=%d",
                    train_report["accuracy"], score, report["auroc"], report["false_positive_rate"],
                    report["recall_fake"], len(seen), len(lookup), state["stale"])
        state.update(epoch=epoch+1, offset=0, plan=None, progress=None)
        state["complete"] = state["stale"] >= args.patience or epoch+1 >= args.epochs
        save_state(run, model, optimizer, scheduler, scaler, state)
        cache.clear_train(epoch)
        if state["complete"]:
            logger.info("Training finished. Early stopping=%s", state["stale"] >= args.patience)
            break
    return model, state
