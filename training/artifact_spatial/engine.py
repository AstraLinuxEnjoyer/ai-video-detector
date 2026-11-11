"""Paired local-patch training with exact durable resume and source-aware selection."""
import json
import csv
import math
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from training.spatial.common import atomic_json, event, fingerprint
from training.spatial.data import Rotation, fixed_selection
from training.spatial.engine import atomic_torch, build_model, loader, seed_all, save_state
from training.spatial.media import FrameCache
from training.spatial.metrics import grouped_report, bootstrap_interval
from .media import prepare
from .objective import hard_weights, paired_loss, selection_score, update_history
from .patches import PatchFrames, aggregate_views
from .sampling import MatchedRotation, distribution, spatial_bucket
from .metadata import attach_dimensions


def is_hybrid(args):
    return getattr(args, "recipe", "patch-v1") == "hybrid-v2"


def is_full_frame(args):
    return getattr(args, "recipe", "patch-v1") == "full-frame-v1"


def write_predictions(path, rows):
    fields = ["id", "path", "group", "source", "label", "score", "max_score", "frames",
              "patch_spread", "global_score", "local_score"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def patch_scores(model, pixels, device, chunk=8):
    shape = pixels.shape[:2]
    flat = pixels.flatten(0, 1)
    outputs = []
    for part in flat.split(chunk):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs.append(model(part.to(device)).float().sigmoid().flatten().cpu())
    scores = torch.cat(outputs).reshape(shape)
    if not torch.isfinite(scores).all():
        raise RuntimeError("Nonfinite patch predictions")
    return scores


def evaluate(model, records, args, device, log, tag, stress=False):
    model.eval()
    scores, spreads = defaultdict(list), defaultdict(list)
    global_scores, local_scores, frame_rows = defaultdict(list), defaultdict(list), []
    hybrid = is_hybrid(args)
    with torch.inference_mode():
        for pixels, labels, ids, sources in tqdm(loader(PatchFrames(records, args.image_size, stress=stress,
                                                                    hybrid=hybrid, full_frame=is_full_frame(args)),
                    args.batch_size, args.workers), desc=tag):
            patches = patch_scores(model, pixels, device)
            for item, row, label, source in zip(ids, patches, labels, sources):
                value = float(aggregate_views(row, hybrid))
                scores[item].append(value)
                spreads[item].append(float(row.max()-row.min()))
                frame_rows.append(dict(id=item, source=source, label=int(label), score=value))
                if hybrid:
                    global_scores[item].append(float(row[0]))
                    local_scores[item].append(float(row[1:].mean()))
    rows = [dict(id=r["id"], path=r["path"], group=r["group"], source=r["source"], label=r["label"],
                 score=float(np.mean(scores[r["id"]])), max_score=max(scores[r["id"]]),
                 patch_spread=float(np.mean(spreads[r["id"]])), frames=len(scores[r["id"]])) for r in records]
    if any(r["frames"] != len(original["frames"]) for r, original in zip(rows, records)):
        raise RuntimeError("Incomplete evaluation coverage")
    report = grouped_report(rows)
    if hybrid:
        by_id = {r["id"]: r for r in records}
        for row in rows:
            row.update(global_score=float(np.mean(global_scores[row["id"]])),
                       local_score=float(np.mean(local_scores[row["id"]])))
        report["global_only"] = grouped_report([dict(r, score=r["global_score"]) for r in rows])
        report["local_only"] = grouped_report([dict(r, score=r["local_score"]) for r in rows])
        report["frame_metrics"] = grouped_report(frame_rows)
        report["group_bootstrap_balanced_accuracy_95ci"] = bootstrap_interval(rows, args.seed)
        buckets = defaultdict(list)
        for row in rows:
            buckets[spatial_bucket(by_id[row["id"]])].append(row)
        report["by_spatial_bucket"] = {key: grouped_report(values) for key, values in buckets.items()}
        report["branch_disagreement"] = sum((r["global_score"] >= .5) != (r["local_score"] >= .5) for r in rows)
        report["largest_errors"] = sorted([r for r in rows if (r["score"] >= .5) != bool(r["label"])],
                                           key=lambda r: abs(r["score"]-r["label"]), reverse=True)[:30]
        log.info("%s global_bAcc=%s local_bAcc=%s frame_bAcc=%s disagreement=%d CI=%s", tag,
                 report["global_only"]["balanced_accuracy"], report["local_only"]["balanced_accuracy"],
                 report["frame_metrics"]["balanced_accuracy"], report["branch_disagreement"],
                 report["group_bootstrap_balanced_accuracy_95ci"])
    report["confident_errors"] = sum((r["label"] == 0 and r["score"] >= .95) or
                                      (r["label"] == 1 and r["score"] <= .05) for r in rows)
    report["mean_patch_spread"] = float(np.mean([r["patch_spread"] for r in rows]))
    log.info("%s n=%d balanced_acc=%s FPR=%s recall_fake=%s confident_errors=%d", tag, report["n"],
             report["balanced_accuracy"], report["false_positive_rate"], report["recall_fake"], report["confident_errors"])
    for source, values in report["by_source"].items():
        log.info("%s SOURCE %s n=%d correct=%d FP=%d FN=%d", tag, source, values["n"],
                 values["tn"]+values["tp"], values["fp"], values["fn"])
    return rows, report


def train_step(model, pixels, labels, weights, optimizer, scaler, device, hybrid=False):
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    use_amp = scaler.is_enabled()
    retries = []
    for attempt in range(3 if use_amp else 1):
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        amp = use_amp and attempt < 2
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(pixels.flatten(0, 2)).reshape(pixels.shape[:3])
            loss, raw, agreement = paired_loss(logits, labels, weights, hybrid=hybrid)
        if not torch.isfinite(loss):
            if amp:
                before = scaler.get_scale()
                scaler.scale(loss)
                scaler.update(new_scale=before/2)
                retries.append(dict(attempt=attempt+1, reason="nonfinite_loss",
                                    scale_before=before, scale_after=scaler.get_scale()))
                continue
            raise RuntimeError("Nonfinite paired loss; batch not advanced")
        if amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if not finite:
            if not amp:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError("Nonfinite gradients in FP32; batch not advanced")
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            retries.append(dict(attempt=attempt+1, scale_before=before, scale_after=scaler.get_scale()))
            continue
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        return dict(loss=float(loss.detach()), raw=raw.cpu().tolist(), agreement=float(agreement.mean()),
                    grad_norm=float(norm), retries=retries,
                    scores=aggregate_views(logits.detach().float().sigmoid().mean(1), hybrid).cpu().tolist())
    raise RuntimeError("No successful optimizer update")


def validation(model, records, args, device, log, run, tag):
    clean_rows, clean = evaluate(model, records, args, device, log, tag)
    stress_rows, stress = evaluate(model, records, args, device, log, tag+" STRESS", stress=True)
    selection = selection_score(clean, stress, args.min_source_count)
    write_predictions(run / f"{tag}_clean.csv", clean_rows)
    write_predictions(run / f"{tag}_stress.csv", stress_rows)
    report = dict(clean=clean, stress=stress, selection=selection)
    atomic_json(run / f"{tag}.json", report)
    log.info("%s selection=%s", tag, selection)
    return report


def train(run, payload, args, log):
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    model = build_model(False).to(device)
    # Freeze running statistics: tiny correlated patch batches otherwise shift BN by source.
    model.classifier[0].p = 0.
    optimizer = torch.optim.AdamW([
        dict(params=model.features.parameters(), lr=args.backbone_lr),
        dict(params=model.classifier.parameters(), lr=args.head_lr)], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.)
    hybrid = is_hybrid(args)
    full_frame = is_full_frame(args)
    matched = hybrid or full_frame
    state = dict(schema=getattr(args, "recipe", "patch-v1"), split_hash=payload["fingerprint"], epoch=0, offset=0, plan=None,
                 rotation=None, seen=[], hard={}, best=-1., stale=0, complete=False, reserve_gb=args.reserve_gb,
                 config_hash=fingerprint(vars(args)))
    checkpoint = run / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if any(saved[key] != state[key] for key in ("schema", "split_hash", "config_hash")):
            raise ValueError("Incompatible checkpoint")
        model.load_state_dict(saved.pop("model"))
        optimizer.load_state_dict(saved.pop("optimizer"))
        scheduler.load_state_dict(saved.pop("scheduler"))
        scaler.load_state_dict(saved.pop("scaler"))
        torch.set_rng_state(saved.pop("torch_rng"))
        random.setstate(saved.pop("python_rng"))
        np.random.set_state(saved.pop("numpy_rng"))
        cuda_rng = saved.pop("cuda_rng")
        if cuda_rng is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng)
        state = saved
    else:
        model.load_state_dict(torch.load(run / "initial.pt", map_location="cpu", weights_only=True))
    log.info("PATCH train device=%s epoch=%d offset=%d config=%s", device, state["epoch"], state["offset"], vars(args))
    if state["complete"]:
        return model, state
    lineage = json.loads((run / "lineage.json").read_text())
    inherited = set(lineage["seen_ids"])
    training_records = payload["splits"]["train"]
    if matched:
        training_records = attach_dimensions(training_records, json.loads((run / "dimensions.json").read_text()))
    lookup = {r["id"]: r for r in training_records}
    if not set(state["hard"]) <= set(lookup):
        raise ValueError("Hard-example history contains non-training IDs")
    rotation_type = MatchedRotation if matched else Rotation
    rotation = rotation_type(list(lookup.values()), args.seed, state["rotation"], previously_seen=inherited)
    if matched:
        atomic_json(run / "sampling_audit.json", rotation.audit)
        log.info("MATCHED SAMPLING cells=%s unmatched_pool=%d UCF cap=20%% of real; no codec/content matching",
                 rotation.cells, len(rotation.audit["unmatched_ids"]))
    cache = FrameCache(run, args.cache_gb, args.reserve_gb)
    for epoch in range(state["epoch"]):
        cache.clear_train(epoch)
    val = prepare(fixed_selection(payload["splits"]["val"], args.val_videos, args.seed), cache, "val",
                  args.val_frames, args.seed, log, run, args.decode_workers)

    def save_best(epoch, report):
        atomic_torch(run / "best.pt", model.state_dict())
        aggregation = ("mean_full_frame_probability" if full_frame else
                       "half_global_half_local_then_mean_frames" if hybrid else "mean_patch_probability_then_mean_frames")
        atomic_json(run / "best.json", dict(epoch=epoch, recipe=state["schema"], image_size=args.image_size,
                    decode_max_side=960, patches=0 if full_frame else 5, aggregation=aggregation,
                    selection=report["selection"], threshold=.5, calibrated=False, split_hash=payload["fingerprint"]))

    if state["best"] < 0:
        # Measure the old full-frame recipe on these SAME videos for a useful A/B reference.
        from training.spatial.engine import evaluate as full_evaluate
        from argparse import Namespace
        parent_args = Namespace(image_size=384, val_batch_size=args.batch_size, workers=args.workers, seed=args.seed)
        # Its Frames resize/center-crop matches the old inference recipe, but decode here is 960px.
        _, reference = full_evaluate(model, val, parent_args, device, log, "RGB REFERENCE 960 DECODE")
        atomic_json(run / "rgb_reference.json", dict(reference, caveat="Same videos; decode=960 not legacy640; diagnostic only"))
        initial = validation(model, val, args, device, log, run, "initial")
        state["best"] = initial["selection"]["score"]
        save_best(0, initial)
        save_state(run, model, optimizer, scheduler, scaler, state)
    for epoch in range(state["epoch"], args.epochs):
        state["epoch"] = epoch
        if state["plan"] is None:
            selected = rotation.take(args.videos_per_epoch)
            state.update(plan=[r["id"] for r in selected], rotation=rotation.state,
                         progress=dict(frames=0, loss_sum=0., correct=0, steps=0, retries=0),
                         epoch_hard=dict(state["hard"]))
            save_state(run, model, optimizer, scheduler, scaler, state)
        selected = [lookup[item] for item in state["plan"]]
        records = prepare(selected, cache, f"train_{epoch}", args.frames_per_video, args.seed+epoch,
                          log, run, args.decode_workers)
        atomic_json(run / f"epoch_{epoch+1:03}_plan.json", dict(ids=state["plan"],
                    sources=dict(Counter(r["source"] for r in selected)), labels=dict(Counter(r["label"] for r in selected)),
                    spatial_buckets=distribution(selected) if matched else None))
        if matched:
            log.info("EPOCH PLAN sources=%s spatial_cells=%s", dict(Counter(r["source"] for r in selected)), distribution(selected))
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        dataset = PatchFrames(records, args.image_size, train=True, seed=args.seed+epoch,
                              hybrid=hybrid, full_frame=full_frame)
        progress = state["progress"]
        loop = tqdm(loader(dataset, args.batch_size, args.workers, state["offset"]),
                    total=math.ceil(len(dataset)/args.batch_size), initial=state["offset"]//args.batch_size,
                    desc=f"PATCH TRAIN {epoch+1}")
        for pixels, labels, ids, sources in loop:
            weights = torch.tensor(hard_weights(ids, sources, state["epoch_hard"]), device=device)
            before = model.classifier[1].weight.detach().clone()
            result = train_step(model, pixels.to(device), labels.float().to(device), weights, optimizer, scaler, device, hybrid=hybrid)
            update_history(state["hard"], ids, sources, result.pop("raw"))
            predictions = result.pop("scores")
            state["seen"] = sorted(set(state["seen"]).union(ids))
            state["offset"] += len(ids)
            progress["frames"] += len(ids)
            progress["correct"] += sum((p >= .5) == bool(y) for p, y in zip(predictions, labels.tolist()))
            progress["loss_sum"] += result["loss"]*len(ids)
            progress["steps"] += 1
            progress["retries"] += len(result["retries"])
            delta = float((model.classifier[1].weight.detach()-before).abs().mean())
            if not math.isfinite(delta):
                raise RuntimeError("Nonfinite parameter update")
            details = dict(result, epoch=epoch+1, step=progress["steps"], frames=state["offset"],
                           classifier_update_mean=delta, hard_weight_mean=float(weights.mean()),
                           learning_rates=[g["lr"] for g in optimizer.param_groups],
                           gpu_peak_gb=torch.cuda.max_memory_allocated()/1024**3 if device.type == "cuda" else 0)
            if result["retries"]:
                event(run / "numerics.jsonl", details)
            if progress["steps"] == 1 or progress["steps"] % args.log_every == 0:
                log.info("PATCH BATCH %s", details)
                event(run / "batches.jsonl", details)
            loop.set_postfix(loss=f"{result['loss']:.4f}", agreement=f"{result['agreement']:.4f}")
            if progress["steps"] % args.save_every == 0:
                save_state(run, model, optimizer, scheduler, scaler, state)
        save_state(run, model, optimizer, scheduler, scaler, state)
        report = validation(model, val, args, device, log, run, f"epoch_{epoch+1:03}")
        score = report["selection"]["score"]
        if score > state["best"]+1e-4:
            state["best"], state["stale"] = score, 0
            save_best(epoch+1, report)
        else:
            state["stale"] += 1
        scheduler.step(score)
        report.update(epoch=epoch+1, train_loss=progress["loss_sum"]/progress["frames"],
                      train_accuracy=progress["correct"]/progress["frames"], retries=progress["retries"],
                      unique_train_seen=len(state["seen"]),
                      unique_seen_including_parent=len(set(state["seen"])|inherited),
                      train_pool=len(lookup), stale=state["stale"])
        atomic_json(run / f"epoch_{epoch+1:03}.json", report)
        event(run / "epochs.jsonl", report)
        seen_set = set(state["seen"])
        atomic_json(run / "coverage.json", dict(seen_ids=state["seen"], by_source={source:
                    dict(pool=sum(r["source"]==source for r in selected_pool),
                         seen=sum(r["source"]==source and r["id"] in seen_set for r in selected_pool))
                    for selected_pool in [list(lookup.values())] for source in sorted({r["source"] for r in selected_pool})}))
        atomic_torch(run / "last.pt", model.state_dict())
        log.info("PATCH EPOCH %d selection=%.4f train=%.4f seen=%d stale=%d", epoch+1, score,
                 report["train_accuracy"], len(state["seen"]), state["stale"])
        state.update(epoch=epoch+1, offset=0, plan=None, complete=state["stale"]>=args.patience or epoch+1>=args.epochs)
        save_state(run, model, optimizer, scheduler, scaler, state)
        cache.clear_train(epoch)
        if state["complete"]:
            break
    return model, state
