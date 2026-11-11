"""Index DFDC in place, preserving the original/fake family as the split unit."""

import hashlib
import json
import os
from pathlib import Path

from training.spatial.common import ROOT, atomic_json, fingerprint

INDEX = ROOT / "cache/datasets/dfdc.json"


def default_root():
    return Path(os.environ.get("AI_VIDEO_DFDC_ROOT", Path.home() / "Downloads" / "dfdc"))


def family_split(origin):
    value = int(hashlib.sha256(f"dfdc-v1:{origin}".encode()).hexdigest()[:12], 16) / 16**12
    return "train" if value < .8 else "val" if value < .9 else "test"


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON key: {name}")
        result[name] = value
    return result


def build_index(root):
    root = Path(root).resolve(strict=True)
    parts = sorted(root.rglob("metadata.json"))
    if not parts:
        raise ValueError(f"No DFDC metadata.json files under {root}")
    records, metadata = [], []
    for path in parts:
        directory = path.parent
        raw = path.read_bytes()
        entries = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Invalid DFDC metadata: {path}")
        files = {p.name: p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"}
        if set(entries) != set(files):
            raise ValueError(f"DFDC metadata/MP4 mismatch in {directory}: "
                             f"{len(set(entries)-set(files))} missing, {len(set(files)-set(entries))} unlabeled")
        if len({name.casefold() for name in files}) != len(files):
            raise ValueError(f"Case-colliding DFDC filenames in {directory}")
        for name, entry in sorted(entries.items()):
            if not isinstance(entry, dict) or entry.get("label") not in ("REAL", "FAKE"):
                raise ValueError(f"Invalid DFDC label: {directory / name}")
            fake = entry["label"] == "FAKE"
            original = entry.get("original") if fake else name
            if (not isinstance(original, str) or original not in entries
                    or not isinstance(entries[original], dict) or entries[original].get("label") != "REAL"):
                raise ValueError(f"DFDC fake without a labeled original in its part: {directory / name}")
            video = files[name].resolve(strict=True)
            if not video.is_relative_to(root):
                raise ValueError(f"DFDC file points outside dataset root: {video}")
            origin = "dfdc/" + (directory / original).relative_to(root).as_posix()
            stat = video.stat()
            records.append(dict(path=str(video), relative=(directory / name).relative_to(root).as_posix(),
                                label=int(fake), source="DFDC", origin=origin, group=origin,
                                partition=family_split(origin), size=stat.st_size, mtime=stat.st_mtime_ns))
        metadata.append(dict(part=directory.relative_to(root).as_posix(), count=len(entries),
                             sha256=hashlib.sha256(raw).hexdigest()))
    return dict(schema=1, root=str(root), metadata=metadata, records=records,
                fingerprint=fingerprint(records))


def write_index(root, output=INDEX):
    index = build_index(root)
    atomic_json(output, index)
    return index
