"""Read-only replay of a few batches after a real checkpoint; never saves production weights."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import random
import torch

from training.spatial.common import ROOT, atomic_json, logger_for
from training.spatial.engine import Frames, build_model, loader, optimizer_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--force-fp32-retry", action="store_true", help="Inject two test-only overflows to exercise FP32 recovery")
    args = parser.parse_args()
    run = Path(json.loads((ROOT / "cache/spatial_v2/latest.json").read_text())["run_dir"])
    checkpoint_path = run / "checkpoint.pt"
    before_stat = checkpoint_path.stat()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = json.loads((run / "config.json").read_text())
    payload = json.loads((run / "splits.json").read_text())
    lookup = {r["id"]: r for r in payload["splits"]["train"]}
    records = []
    for item in saved["plan"]:
        frames = json.loads((run / "frames" / f"train_{saved['epoch']}" / item / "frames.json").read_text())
        records.append(dict(lookup[item], **frames))
    records.sort(key=lambda r: r["id"])
    output = ROOT / "cache/spatial_v2/numerical_checks" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    logger = logger_for(output)
    model = build_model(False).cuda()
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW([
        dict(params=model.features.parameters(), lr=config["backbone_lr"]),
        dict(params=model.classifier.parameters(), lr=config["head_lr"])], weight_decay=1e-4)
    optimizer.load_state_dict(saved["optimizer"])
    scaler = torch.amp.GradScaler("cuda")
    scaler.load_state_dict(saved["scaler"])
    model.train()
    if args.force_fp32_retry:
        failures_left = 2

        def injected_overflow(gradient):
            nonlocal failures_left
            if failures_left:
                failures_left -= 1
                return torch.full_like(gradient, float("inf"))
            return gradient

        model.classifier[1].weight.register_hook(injected_overflow)
        logger.warning("TEST ONLY: injecting two overflows to exercise FP32 fallback")
    if saved["epoch"] < config["warmup_epochs"]:
        model.features.requires_grad_(False)
        model.features.eval()
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    dataset = Frames(records, config["image_size"], train=True, seed=config["seed"] + saved["epoch"])
    rows = []
    for index, (images, labels, ids, frame_indices) in enumerate(loader(
            dataset, config["batch_size"], config["workers"], saved["offset"])):
        loss, logits, norm, retries = optimizer_batch(model, images.cuda(), labels.float().cuda().unsqueeze(1),
            optimizer, scaler, logger, output, dict(epoch=saved["epoch"]+1,
                step=saved["progress"]["steps"]+index+1, video_ids=list(ids), frame_indices=frame_indices.tolist()))
        row = dict(step=saved["progress"]["steps"]+index+1, loss=float(loss), grad_norm=float(norm),
                   retries=retries, scale=scaler.get_scale())
        rows.append(row)
        logger.info("REPLAY %s", row)
        if len(rows) >= args.batches:
            break
    after_stat = checkpoint_path.stat()
    assert (before_stat.st_size, before_stat.st_mtime_ns) == (after_stat.st_size, after_stat.st_mtime_ns)
    atomic_json(output / "report.json", dict(checkpoint=str(checkpoint_path), checkpoint_untouched=True,
                injected_overflows=args.force_fp32_retry, batches=rows,
                gpu_peak_gb=torch.cuda.max_memory_allocated()/1024**3))
    print("REPLAY_OK", output)


if __name__ == "__main__":
    main()
