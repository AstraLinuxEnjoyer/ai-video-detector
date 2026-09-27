"""GPU integration check on mixed DAVIS frames/videos; no production weights modified."""

import argparse
import json
from datetime import datetime

import torch

from training.spatial.common import ROOT, SCHEMA, atomic_json, fingerprint, logger_for
from training.spatial.engine import train
from training.spatial.run import DEFAULTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    options = parser.parse_args()
    from pathlib import Path
    prepared = Path(options.run_dir)
    payload = json.loads((prepared / "splits.json").read_text(encoding="utf-8"))
    config = json.loads((prepared / "config.json").read_text())
    lineage = json.loads((prepared / "lineage.json").read_text())
    sources = {"DAVIS", "KoNViD_1k_videos", "VideoUFO", "OpenAI_Sora", "keling", "GenVidBench:ms"}
    splits = {name: [] for name in ("train", "val", "test", "ood")}
    for split in ("train", "val"):
        for source in sorted(sources):
            splits[split].append(next(r for r in payload["splits"][split] if r["source"] == source))
    root = ROOT / "cache/spatial_v2/smoke" / ("finetune_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    root.mkdir(parents=True)
    atomic_json(root / "lineage.json", lineage)
    args = argparse.Namespace(**{**DEFAULTS, **config, "epochs": 1, "videos_per_epoch": 6,
        "val_videos": 6, "val_frames": 4, "warmup_epochs": 0, "save_every": 1, "log_every": 1})
    atomic_json(root / "config.json", vars(args))
    sample = dict(schema=SCHEMA, splits=splits, fingerprint=fingerprint(splits))
    atomic_json(root / "splits.json", sample)
    log = logger_for(root)
    log.warning("INTEGRATION TEST ONLY: six train/six validation clips; not a quality estimate")
    model, state = train(root, sample, args, log)
    assert state["complete"] and len(state["seen"]) == 6
    assert all(torch.isfinite(p).all() for p in model.parameters())
    print("FINETUNE_SMOKE_OK", root)


if __name__ == "__main__":
    main()
