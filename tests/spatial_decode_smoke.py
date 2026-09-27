"""Decode representative files from every audited source, without training."""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from training.spatial.common import ROOT, atomic_json, logger_for
from training.spatial.media import FrameCache, prepare_frames


def main():
    latest = json.loads((ROOT / "cache/spatial_v2/latest.json").read_text())
    payload = json.loads((Path(latest["run_dir"]) / "splits.json").read_text())
    sources = defaultdict(list)
    for rows in payload["splits"].values():
        for record in rows:
            sources[(record["label"], record["source"])].append(record)
    records = [r for rows in sources.values() for r in sorted(rows, key=lambda r: r["id"])[:2]]
    run = ROOT / "cache/spatial_v2/decode_checks" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    logger = logger_for(run)
    prepared, failures = prepare_frames(records, FrameCache(run, 1, 10), "sources", 3, 42, logger, run, 2)
    atomic_json(run / "report.json", dict(sources=len(sources), requested=len(records),
                decoded=len(prepared), failures=failures,
                caveat="Two videos per source only; not a complete corpus decoder audit"))
    print("DECODE_CHECK", run, "sources", len(sources), "videos", len(prepared), "failed", len(failures))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
