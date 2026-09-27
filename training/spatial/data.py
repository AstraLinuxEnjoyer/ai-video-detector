"""Content deduplication, origin grouping, frozen splits and rotating quotas."""

import hashlib
import json
import random
import re
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from .common import ROOT, SCHEMA, atomic_json, event, fingerprint
from .sequences import audit_sequence, collect_davis
from .lineage import load_parent, pin_parent_splits

EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DEEPACTION = {"Pexels", "BDAnimateDiffLightning", "CogVideoX5B", "RunwayML",
              "StableDiffusion", "Veo", "VideoPoet"}


def identify(relative):
    parts = Path(relative).parts
    source = parts[0] if len(parts) > 1 else "unclassified"
    stem = Path(relative).stem
    if source in DEEPACTION and len(parts) >= 3:
        return "DeepAction:" + source, "deepaction/action/" + parts[1]
    if source == "VideoUFO":
        return source, "videoufo/" + re.sub(r"\.\d+$", "", stem)
    if source == "UCF-101":
        match = re.match(r"(v_.+_g\d+)_c\d+$", stem)
        return source, "ucf/" + (match[1] if match else stem)
    if source in {"ms", "pika", "vc2", "t2vz"}:
        # These four releases share prompt UUIDs across generators.
        return "GenVidBench:" + source, "vidprom/" + stem.split("-", 1)[-1]
    if source.startswith("GenVideo_"):
        # Conservative grouping: matching original numeric IDs stay together.
        return source, "genvideo/" + stem.rsplit("_", 1)[-1]
    return source, "local/" + str(relative).replace("\\", "/")


def expected_source_label(source):
    if source in {"DeepAction:Pexels", "UCF-101", "VideoUFO", "DAVIS", "KoNViD_1k_videos", "KoNViD_1k"}:
        return 0
    if (source.startswith("DeepAction:") or source.startswith("GenVideo_")
            or source.startswith("GenVidBench:")):
        return 1
    return None


def collect_paths(root=ROOT):
    records = []
    for label, name in enumerate(("real", "fake")):
        base = root / "data" / name
        if not base.exists():
            continue
        records.extend(collect_davis(base, label))
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                relative = path.relative_to(base)
                source, group = identify(relative)
                records.append(dict(path=str(path.resolve()), label=label, source=source,
                                    origin=group, relative=str(relative).replace("\\", "/")))
    return records


def remote_paths(records, run, logger, download_gb=20, reserve_gb=10, offline=False):
    """Reuse existing cache; only download missing DeepAction objects, before training."""
    catalog = ROOT / "cache/spatial/catalog.json"
    rows = json.loads(catalog.read_text(encoding="utf-8")) if catalog.exists() else []
    if not rows and not offline:
        try:
            from huggingface_hub import HfApi
            rows = [{"repo_path": p, "repo_id": "faridlab/deepaction_v1"}
                    for p in HfApi().list_repo_files("faridlab/deepaction_v1", repo_type="dataset")]
        except Exception as exc:
            logger.warning("Remote catalog unavailable; scanning existing cache: %s", exc)
            event(run / "data_errors.jsonl", dict(stage="remote_catalog", error=str(exc)))
    existing = {r["relative"] for r in records}
    cache = ROOT / "cache/spatial/videos/faridlab__deepaction_v1"
    cache.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"repo_path": str(p.relative_to(cache)).replace("\\", "/")}
                for p in cache.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS]
    added = 0
    consecutive_errors = 0
    deferred = 0
    for row in tqdm(rows, desc="REMOTE", leave=False):
        if row.get("local_path"):
            continue
        repo_path = row["repo_path"]
        parts = Path(repo_path).parts
        if len(parts) != 3 or parts[0] not in DEEPACTION or Path(repo_path).suffix.lower() not in EXTENSIONS:
            continue
        if row.get("repo_id", "faridlab/deepaction_v1") != "faridlab/deepaction_v1":
            continue
        if repo_path in existing:
            continue
        path = cache / repo_path
        legacy = ROOT / "cache/spatial/videos" / repo_path
        if not path.exists() and legacy.exists():
            path = legacy
        if not path.exists() and not offline and consecutive_errors < 3:
            try:
                from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url
                url = hf_hub_url("faridlab/deepaction_v1", repo_path, repo_type="dataset")
                metadata = get_hf_file_metadata(url, timeout=20)
                size = metadata.size
                if size is None or added + size > download_gb * 1024**3:
                    raise RuntimeError("download budget reached or unknown file size")
                if shutil.disk_usage(cache).free - size < reserve_gb * 1024**3:
                    raise RuntimeError("free disk reserve would be exceeded")
                for attempt in range(3):
                    try:
                        path = Path(hf_hub_download("faridlab/deepaction_v1", repo_path,
                                                   repo_type="dataset", local_dir=str(cache)))
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(1 + attempt)
                added += path.stat().st_size
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                event(run / "data_errors.jsonl", dict(stage="download", path=repo_path, error=str(exc)))
                logger.warning("Remote unavailable: %s: %s", repo_path, exc)
                continue
        if not path.exists():
            deferred += 1
            event(run / "data_errors.jsonl", dict(stage="download_deferred", path=repo_path,
                                                  error="offline or download circuit breaker"))
        if path.exists():
            source, group = identify(repo_path)
            records.append(dict(path=str(path.resolve()), label=int(parts[0] != "Pexels"),
                                source=source, origin=group, relative=repo_path))
    logger.info("Remote cache added %.3f GiB; training will use local files only", added / 1024**3)
    if deferred:
        logger.warning("Remote unavailable/deferred files=%d; only successfully available files enter the snapshot", deferred)
    return records


class UnionFind:
    def __init__(self):
        self.parents = {}

    def find(self, item):
        self.parents.setdefault(item, item)
        node = item
        while self.parents[node] != node:
            node = self.parents[node]
        while self.parents[item] != item:
            parent = self.parents[item]
            self.parents[item] = node
            item = parent
        return node

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parents[max(a, b)] = min(a, b)


def deduplicate(records, database, logger, errors):
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE IF NOT EXISTS hashes (path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, sha TEXT)")
    groups = UnionFind()
    by_hash = defaultdict(list)
    rejected = 0

    def hash_file(path, stat):
        cached = connection.execute("SELECT size,mtime,sha FROM hashes WHERE path=?", (str(path),)).fetchone()
        if cached and cached[:2] == (stat.st_size, stat.st_mtime_ns):
            return cached[2]
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        connection.execute("INSERT OR REPLACE INTO hashes VALUES(?,?,?,?)",
                           (str(path), stat.st_size, stat.st_mtime_ns, sha))
        return sha

    try:
        for index, record in enumerate(tqdm(records, desc="AUDIT SHA256")):
            path = Path(record["path"])
            try:
                expected = expected_source_label(record["source"])
                if expected is not None and record["label"] != expected:
                    raise ValueError(f"Source label mismatch: {record['source']} expects {expected}, folder says {record['label']}; quarantined, not relabeled")
                if record.get("kind") == "image_sequence":
                    record = audit_sequence(record, hash_file)
                    by_hash[record["id"]].append(record)
                    groups.find(record["origin"])
                    connection.commit()
                    continue
                stat = path.stat()
                if stat.st_size < 1024:
                    raise ValueError("file smaller than 1024 bytes (possibly a pointer)")
                with path.open("rb") as stream:
                    header = stream.read(12)
                if (header.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a"))
                        or (header.startswith(b"RIFF") and header[8:12] == b"WEBP")):
                    raise ValueError("Image content disguised as a video container; not a video training sample")
                sha = hash_file(path, stat)
                record = dict(record, id=sha, size=stat.st_size, mtime=stat.st_mtime_ns)
                by_hash[sha].append(record)
                groups.find(record["origin"])
                if index % 100 == 0:
                    connection.commit()
                if index % 1000 == 0:
                    logger.info("Hash audit %d/%d", index + 1, len(records))
            except (OSError, ValueError) as exc:
                rejected += 1
                event(errors, dict(stage="hash", path=str(path), error=str(exc)))
        connection.commit()
    finally:
        connection.close()
    conflicts = []
    for sha, copies in by_hash.items():
        if len({r["label"] for r in copies}) != 1:
            conflicts.append(dict(sha=sha, files=copies))
        for record in copies[1:]:
            groups.union(copies[0]["origin"], record["origin"])
    if conflicts:
        atomic_json(errors.parent / "label_conflicts.json", conflicts)
        raise ValueError("Identical files have conflicting real/fake labels; see label_conflicts.json")
    unique = []
    for sha, copies in sorted(by_hash.items()):
        chosen = min(copies, key=lambda r: ("_downloaded" in r["source"], r["path"]))
        unique.append(dict(chosen, group=groups.find(chosen["origin"]),
                           aliases=[dict(path=r["path"], source=r["source"]) for r in copies]))
    logger.info("Audit records=%d unique=%d duplicates=%d rejected=%d",
                len(records), len(unique), sum(len(v)-1 for v in by_hash.values()), rejected)
    return unique


def make_splits(records, seed=42, holdout="GenVideo_OpenSora"):
    holdout_groups = {r["group"] for r in records
                      if holdout and any(a["source"] == holdout for a in r["aliases"])}
    result = {name: [] for name in ("train", "val", "test", "ood")}
    for record in records:
        score = int(hashlib.sha256(f"{seed}:{record['group']}".encode()).hexdigest()[:12], 16) / 16**12
        split = "ood" if record["group"] in holdout_groups else (
            "train" if score < .8 else "val" if score < .9 else "test")
        result[split].append(record)
    assert_disjoint(result)
    for name in ("train", "val", "test"):
        if {r["label"] for r in result[name]} != {0, 1}:
            raise ValueError(f"{name} needs independent groups from both classes")
    return result


def assert_disjoint(splits):
    for field in ("id", "group"):
        seen = {}
        for split, records in splits.items():
            for record in records:
                key = record[field]
                if key in seen and seen[key] != split:
                    raise ValueError(f"Leakage: shared {field} across {seen[key]} and {split}")
                seen[key] = split


class Rotation:
    """Balance labels/sources; visit every source member before cycling it."""
    def __init__(self, records, seed=42, state=None, previously_seen=None):
        self.records = {r["id"]: r for r in records}
        self.pools = defaultdict(list)
        for record in records:
            self.pools[f"{record['label']}|{record['source']}"].append(record["id"])
        self.seed = seed
        self.previously_seen = set(previously_seen or [])
        self.state = state or {"positions": {}, "turns": {"0": 0, "1": 0}}

    def take(self, count):
        selected, used = [], set()
        for label in (0, 1):
            keys = sorted(k for k in self.pools if k.startswith(f"{label}|"))
            target = min(count // 2, sum(len(self.pools[k]) for k in keys))
            taken = 0
            while taken < target:
                turn = self.state["turns"][str(label)]
                key = keys[turn % len(keys)]
                self.state["turns"][str(label)] += 1
                pool = self.pools[key]
                if all(i in used for i in pool):
                    continue
                position = self.state["positions"].get(key, 0)
                cycle, offset = divmod(position, len(pool))
                queue = sorted(pool)
                random.Random(f"{self.seed}:{key}:{cycle}").shuffle(queue)
                if cycle == 0 and self.previously_seen:
                    queue.sort(key=lambda item: item in self.previously_seen)
                item = queue[offset]
                self.state["positions"][key] = position + 1
                if item in used:
                    continue
                used.add(item)
                selected.append(self.records[item])
                taken += 1
        random.Random(f"{self.seed}:{self.state}").shuffle(selected)
        return selected


def fixed_selection(records, count, seed):
    if {r["label"] for r in records} == {0, 1}:
        return Rotation(records, seed).take(count)
    # OOD may contain only fake videos; keep its metrics explicitly one-class.
    result = sorted(records, key=lambda r: fingerprint([seed, r["id"]]))
    return result[:count]


def prepare(run, args, logger):
    snapshot = run / "splits.json"
    if snapshot.exists():
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        if payload["schema"] != SCHEMA or fingerprint(payload["splits"]) != payload["fingerprint"]:
            raise ValueError("Invalid split snapshot")
        assert_disjoint(payload["splits"])
        return payload
    records = collect_paths()
    logger.info("Found %d local video paths", len(records))
    records = remote_paths(records, run, logger, args.download_gb, args.reserve_gb, args.offline)
    records = deduplicate(records, ROOT / "cache/spatial_v2/hashes.sqlite", logger, run / "data_errors.jsonl")
    splits = make_splits(records, args.seed, args.holdout_source)
    if getattr(args, "parent_run", None):
        lineage, parent = load_parent(args.parent_run)
        splits = pin_parent_splits(splits, parent)
        assert_disjoint(splits)
        atomic_json(run / "lineage.json", lineage)
        logger.info("Fine-tune parent epoch=%d previously seen=%d; old split assignments preserved",
                    lineage["parent_epoch"], len(lineage["seen_ids"]))
    payload = dict(schema=SCHEMA, fingerprint=fingerprint(splits), splits=splits)
    atomic_json(snapshot, payload)
    summary = {}
    for split, values in splits.items():
        summary[split] = dict(videos=len(values), frame_sequences=sum(r.get("kind") == "image_sequence" for r in values), labels=dict(Counter(r["label"] for r in values)),
                              sources=dict(Counter(r["source"] for r in values)))
        logger.info("Split %s: %s", split, summary[split])
    atomic_json(run / "data_summary.json", dict(splits=summary, fingerprint=payload["fingerprint"],
                                               exact_hash_overlap=0, origin_group_overlap=0,
                                               near_duplicate_check="Not exhaustive; known origins plus exact hashes"))
    return payload
