import json
import random
from pathlib import Path

from training.config import (
    RANDOM_SEED,
    SOURCE_MAX_VIDEOS,
    TRAIN_MANIFEST,
    TRAIN_RATIO,
    VAL_MANIFEST,
)
from training.data.catalog import VideoEntry


def _balance_entries(
    real: list[VideoEntry],
    fake: list[VideoEntry],
    rng: random.Random,
) -> tuple[list[VideoEntry], list[VideoEntry]]:
    real = _apply_source_limits(real, rng)
    fake = _apply_source_limits(fake, rng)

    min_count = min(len(real), len(fake))

    if min_count == 0:
        raise RuntimeError("Need both real and fake videos in the catalog.")

    return (
        _source_balanced_sample(real, min_count, rng),
        _source_balanced_sample(fake, min_count, rng),
    )


def _apply_source_limits(entries: list[VideoEntry], rng: random.Random) -> list[VideoEntry]:
    by_source: dict[str, list[VideoEntry]] = {}
    for entry in entries:
        by_source.setdefault(entry.source, []).append(entry)

    limited: list[VideoEntry] = []

    for source, source_entries in by_source.items():
        source_entries = list(source_entries)
        rng.shuffle(source_entries)

        max_videos = SOURCE_MAX_VIDEOS.get(source)
        if max_videos is not None:
            source_entries = source_entries[: min(max_videos, len(source_entries))]

        limited.extend(source_entries)

    return limited


def _source_balanced_sample(
    entries: list[VideoEntry],
    count: int,
    rng: random.Random,
) -> list[VideoEntry]:
    if len(entries) <= count:
        result = list(entries)
        rng.shuffle(result)
        return result

    by_source: dict[str, list[VideoEntry]] = {}
    for entry in entries:
        by_source.setdefault(entry.source, []).append(entry)

    queues: list[list[VideoEntry]] = []
    for source_entries in by_source.values():
        source_entries = list(source_entries)
        rng.shuffle(source_entries)
        queues.append(source_entries)

    rng.shuffle(queues)

    selected: list[VideoEntry] = []
    while len(selected) < count and queues:
        next_queues: list[list[VideoEntry]] = []

        for queue in queues:
            if len(selected) >= count:
                break

            if queue:
                selected.append(queue.pop())

            if queue:
                next_queues.append(queue)

        queues = next_queues
        rng.shuffle(queues)

    rng.shuffle(selected)
    return selected


def _split_class_by_group(
    entries: list[VideoEntry],
    train_ratio: float,
    rng: random.Random,
) -> tuple[list[VideoEntry], list[VideoEntry]]:
    groups: dict[str, list[VideoEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.group_id, []).append(entry)

    grouped_entries = list(groups.values())
    rng.shuffle(grouped_entries)

    train: list[VideoEntry] = []
    val: list[VideoEntry] = []
    train_target = int(len(entries) * train_ratio)

    for group in grouped_entries:
        if len(train) < train_target:
            train.extend(group)
        else:
            val.extend(group)

    if not val and len(grouped_entries) > 1:
        moved = grouped_entries[-1]
        for entry in moved:
            train.remove(entry)
        val.extend(moved)

    return train, val


def _balanced_split(entries: list[VideoEntry], rng: random.Random) -> list[VideoEntry]:
    real = [entry for entry in entries if entry.is_real]
    fake = [entry for entry in entries if not entry.is_real]
    real, fake = _balance_entries(real, fake, rng)

    combined = [*real, *fake]
    rng.shuffle(combined)
    return combined


def split_entries(
    entries: list[VideoEntry],
    train_ratio: float = TRAIN_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple[list[VideoEntry], list[VideoEntry]]:
    rng = random.Random(seed)

    real = [entry for entry in entries if entry.is_real]
    fake = [entry for entry in entries if not entry.is_real]

    real_train, real_val = _split_class_by_group(real, train_ratio, rng)
    fake_train, fake_val = _split_class_by_group(fake, train_ratio, rng)

    train_entries = _balanced_split([*real_train, *fake_train], rng)
    val_entries = _balanced_split([*real_val, *fake_val], rng)

    if not train_entries or not val_entries:
        raise RuntimeError(
            "Train/val split produced an empty split. "
            "Try lowering TRAIN_RATIO or check the catalog size."
        )

    return train_entries, val_entries


def _entry_to_manifest(entry: VideoEntry) -> dict:
    return {
        "id": entry.id,
        "repo_path": entry.repo_path,
        "label": entry.label,
        "source": entry.source,
        "group_id": entry.group_id,
        "local_path": entry.local_path,
        "repo_id": entry.repo_id,
        "repo_type": entry.repo_type,
        "dataset": entry.dataset,
    }


def save_split_manifests(
    train_entries: list[VideoEntry],
    val_entries: list[VideoEntry],
) -> None:
    TRAIN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    train_payload = [_entry_to_manifest(entry) for entry in train_entries]
    val_payload = [_entry_to_manifest(entry) for entry in val_entries]

    TRAIN_MANIFEST.write_text(
        json.dumps(train_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    VAL_MANIFEST.write_text(
        json.dumps(val_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_split_manifest(path: Path) -> list[VideoEntry]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [VideoEntry(**item) for item in raw]
