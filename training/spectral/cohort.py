"""Build mixed-source cohorts without changing the parent splits."""

import json
import random
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

from training.spatial.common import fingerprint
from training.spatial.data import assert_disjoint, fixed_selection
from training.spatial.lineage import file_sha
from .features import verify_record


def paired_selection(rows, pair_count, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    families = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(families)
    selected = []
    for group in families:
        real = [r for r in groups[group] if r["label"] == 0]
        fake = sorted((r for r in groups[group] if r["label"] == 1), key=lambda r: r["relative"])
        if len(real) != 1 or not fake:
            continue
        if len(selected) // 2 >= pair_count:
            break
        selected.extend((real[0], rng.choice(fake)))
    return selected


def hash_records(rows, database=None, publish=None):
    def calculate(connection):
        for step, row in enumerate(rows, 1):
            verify_record(row)
            cached = connection.execute("SELECT size,mtime,sha FROM hashes WHERE path=?",
                                        (row["path"],)).fetchone() if connection else None
            if cached and cached[:2] == (row["size"], row["mtime"]):
                sha = cached[2]
            else:
                sha = file_sha(Path(row["path"]))
                verify_record(row)
                if connection:
                    connection.execute("INSERT OR REPLACE INTO hashes VALUES(?,?,?,?)",
                                       (row["path"], row["size"], row["mtime"], sha))
            if connection and step % 32 == 0:
                connection.commit()
            if publish and (step % 50 == 0 or step == len(rows)):
                publish(completed=step, total=len(rows), detail="dfdc_hashes")
            yield dict(row, id=sha)
        if connection:
            connection.commit()

    if database is None:
        return list(calculate(None))
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database, timeout=60)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS hashes (path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, sha TEXT)")
        return list(calculate(connection))


def build_cohort(parent, dfdc, counts, fraction, seed, *, hash_database=None, publish=None):
    payload = json.loads(Path(parent).read_text(encoding="utf-8"))
    base = payload["splits"]
    if fingerprint(base) != payload["fingerprint"]:
        raise ValueError("Parent split fingerprint mismatch")
    assert_disjoint(base)
    old_ids = {r["id"] for rows in base.values() for r in rows}
    all_train = counts["train"] == 0
    splits, planned = {}, {}
    for offset, (split, count) in enumerate(counts.items()):
        split_seed = seed + offset
        candidates = [r for r in dfdc["records"] if r["partition"] == split]
        full = split == "train" and all_train
        selected = candidates if full else paired_selection(candidates, int(count * fraction / 2), split_seed)
        if split != "ood" and not selected:
            raise ValueError(f"No DFDC pairs selected for {split}; increase the sample count or quota")
        old = [r for r in base[split] if r.get("kind") != "image_sequence"]
        if split != "ood" and {r["label"] for r in old} != {0, 1}:
            raise ValueError(f"{split} needs old and DFDC videos of both classes")
        splits[split] = old if full else fixed_selection(old, count - len(selected), split_seed)
        planned[split] = selected
    selected = [r for rows in planned.values() for r in rows]
    chosen = {r["relative"] for r in selected}
    hashed = hash_records(dfdc["records"] if all_train else selected, hash_database, publish)
    owners, quarantined = {}, set()
    for row in hashed:
        sha = row["id"]
        if sha in old_ids or sha in owners:
            if not all_train:
                raise ValueError(f"Selected DFDC duplicate requires review before training: {row['path']}")
            quarantined.add(row["group"])
            if sha in owners:
                quarantined.add(owners[sha])
        owners[sha] = row["group"]
    for row in hashed:
        if row["relative"] in chosen and row["group"] not in quarantined:
            splits[row["partition"]].append(row)
    excluded = [dict(group=row["group"], path=row["path"], partition=row["partition"])
                for row in hashed if row["group"] in quarantined]
    if all_train:
        suspect = "figure status update - openai speech-to-speech reasoning.mp4"
        known = [r for r in splits["train"] if r["label"] == 1 and Path(r["path"]).name.lower() == suspect]
        ids = {r["id"] for r in known}
        splits["train"] = [r for r in splits["train"] if r["id"] not in ids]
    else:
        known = []
    splits = {split: sorted(rows, key=lambda r: r["id"]) for split, rows in splits.items()}
    for split in ("train", "val", "test"):
        if {r["label"] for r in splits[split] if r["source"] == "DFDC"} != {0, 1}:
            raise ValueError(f"{split} needs both DFDC classes after duplicate checks")
    assert_disjoint(splits)
    return dict(parent_fingerprint=payload["fingerprint"], dfdc_fingerprint=dfdc["fingerprint"],
                fingerprint=fingerprint(splits), splits=splits, dfdc_duplicate_exclusions=excluded,
                known_label_exclusions=[r["path"] for r in known])
