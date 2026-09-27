"""Warm-start provenance and immutable split assignments across experiments."""

import hashlib
import json
from pathlib import Path

from .common import fingerprint


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_parent(directory):
    directory = Path(directory).resolve()
    payload = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
    best = json.loads((directory / "best.json").read_text(encoding="utf-8"))
    if fingerprint(payload["splits"]) != payload["fingerprint"] or best["split_hash"] != payload["fingerprint"]:
        raise ValueError("Parent best weights and dataset snapshot disagree")
    seen = set()
    ancestry = directory / "lineage.json"
    if ancestry.exists():
        seen.update(json.loads(ancestry.read_text(encoding="utf-8"))["seen_ids"])
    for epoch in range(1, best["epoch"] + 1):
        seen.update(json.loads((directory / f"epoch_{epoch:03}_plan.json").read_text())["ids"])
    return dict(parent_run=str(directory), parent_epoch=best["epoch"],
                parent_split_hash=payload["fingerprint"],
                weights_sha256=file_sha(directory / "best.pt"), seen_ids=sorted(seen)), payload["splits"]


def pin_parent_splits(splits, parent):
    by_id, by_origin, by_path = {}, {}, {}
    for split, records in parent.items():
        for record in records:
            by_id[record["id"]] = (split, record["label"])
            for key in (record["group"], record.get("origin", record["group"])):
                by_origin.setdefault(key, set()).add(split)
            for alias in record["aliases"]:
                if "path" in alias:
                    by_path.setdefault(alias["path"], set()).add(split)
    records = [r for rows in splits.values() for r in rows]
    pinned = {}
    for record in records:
        targets = set()
        if record["id"] in by_id:
            split, label = by_id[record["id"]]
            if label != record["label"]:
                raise ValueError("Parent sample label changed")
            targets.add(split)
        for key in (record["group"], record.get("origin", record["group"])):
            targets.update(by_origin.get(key, set()))
        for alias in record["aliases"]:
            targets.update(by_path.get(alias.get("path"), set()))
        pinned.setdefault(record["group"], set()).update(targets)
    if any(len(targets) > 1 for targets in pinned.values()):
        raise ValueError("New duplicates connect different parent splits; resolve leakage before fine-tuning")
    result = {name: [] for name in splits}
    for default, rows in splits.items():
        for record in rows:
            targets = pinned[record["group"]]
            result[next(iter(targets)) if targets else default].append(record)
    return result
