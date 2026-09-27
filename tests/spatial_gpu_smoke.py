"""Small real-video/GPU integration check; never used as model quality evidence."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.spatial.data import identify
from training.spatial.engine import train
from training.spatial.run import DEFAULTS, final_evaluation


def main():
    root = ROOT / "cache/spatial_v2/smoke" / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True)
    catalog = json.loads((ROOT / "cache/spatial/catalog.json").read_text())
    chosen = []
    for label, source in [(0, "local:VideoUFO"), (1, "local:GenVideo_pika")]:
        candidates = [r for r in catalog if r["label"] == label and r["source"] == source]
        for r in candidates[:6]:
            path = ROOT / r["local_path"]
            stat = path.stat()
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            canonical, group = identify(Path(r["local_path"]).relative_to(f"data/{'real' if label == 0 else 'fake'}"))
            chosen.append(dict(id=sha, path=str(path), size=stat.st_size, mtime=stat.st_mtime_ns,
                               source=canonical, group=group, label=label))
    splits = {"train": [], "val": [], "test": [], "ood": []}
    for label in (0, 1):
        rows = [r for r in chosen if r["label"] == label]
        for i, split in enumerate(("train", "val", "test")):
            splits[split].extend(rows[2*i:2*i+2])
    args = argparse.Namespace(**{**DEFAULTS, "epochs": 2, "videos_per_epoch": 4,
        "frames_per_video": 2, "val_frames": 2, "val_videos": 4, "test_videos": 4,
        "batch_size": 8, "val_batch_size": 8, "workers": 2, "robustness_videos": 2,
        "log_every": 1, "save_every": 1})
    atomic_json(root / "config.json", vars(args))
    payload = dict(splits=splits, fingerprint=fingerprint(splits))
    atomic_json(root / "splits.json", payload)
    log = logger_for(root)
    log.warning("SMOKE TEST ONLY: 12 videos, not a quality estimate or production dataset")
    model, state = train(root, payload, args, log)
    assert state["complete"]
    del model
    final_evaluation(root, payload, args, log)
    print("SMOKE_OK", root)


if __name__ == "__main__":
    main()
