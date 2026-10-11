"""Real-file patch training smoke check; writes only an isolated smoke run."""
import argparse
import json
import shutil
from datetime import datetime

import torch
from filelock import FileLock

from training.spatial.common import ROOT, atomic_json, fingerprint, logger_for
from training.artifact_spatial.engine import train


def main():
    prepared = ROOT / "cache/artifact_spatial/runs/patch_v1"
    with FileLock(str(ROOT / "cache/spatial_v2/training.lock"), timeout=0):
        payload = json.loads((prepared / "splits.json").read_text(encoding="utf-8"))
        config = json.loads((prepared / "config.json").read_text())
        sources = ["DAVIS", "KoNViD_1k_videos", "VideoUFO", "OpenAI_Sora", "keling", "GenVidBench:ms"]
        splits = {key: [] for key in payload["splits"]}
        for split in ("train", "val"):
            splits[split] = [next(r for r in payload["splits"][split] if r["source"] == source) for source in sources]
        run = ROOT / "cache/artifact_spatial/smoke" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run.mkdir(parents=True)
        shutil.copy2(prepared / "initial.pt", run / "initial.pt")
        shutil.copy2(prepared / "lineage.json", run / "lineage.json")
        args = argparse.Namespace(**{**config, "epochs": 1, "videos_per_epoch": 6, "val_videos": 6,
                                     "val_frames": 2, "min_source_count": 1, "log_every": 1, "save_every": 1})
        atomic_json(run / "config.json", vars(args))
        snapshot = dict(splits=splits, fingerprint=fingerprint(splits))
        atomic_json(run / "splits.json", snapshot)
        log = logger_for(run)
        log.warning("SMOKE ONLY: six train/six val, not a population quality estimate")
        model, state = train(run, snapshot, args, log)
        assert state["complete"] and len(state["seen"]) == 6
        assert all(torch.isfinite(p).all() for p in model.parameters())
        initial = torch.load(run / "initial.pt", weights_only=True, map_location="cpu")
        assert not torch.equal(initial["classifier.1.weight"], model.classifier[1].weight.detach().cpu())
        for name, value in model.state_dict().items():
            if "running_" in name or "num_batches_tracked" in name:
                torch.testing.assert_close(value.cpu(), initial[name], rtol=0, atol=0)
        print("ARTIFACT_GPU_SMOKE_OK", run)


if __name__ == "__main__":
    main()
