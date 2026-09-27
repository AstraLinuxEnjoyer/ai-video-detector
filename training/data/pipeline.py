import json
import random
from pathlib import Path

from training.config import (
    RANDOM_SEED,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
    VIDEOS_PER_EPOCH,
)
from training.data.catalog import VideoEntry, ensure_catalog, load_catalog
from training.data.splits import (
    load_split_manifest,
    save_split_manifests,
    split_entries,
)


def _manifest_is_valid(path: Path) -> bool:
    if not path.exists():
        return False

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        return False

    required = {"id", "repo_path", "label", "source", "group_id", "dataset", "repo_id"}
    return required.issubset(raw[0])


def prepare_training_manifests(
    refresh_catalog: bool = False,
    rebuild: bool = False,
) -> tuple[list[VideoEntry], list[VideoEntry]]:
    manifests_ready = (
        _manifest_is_valid(TRAIN_MANIFEST)
        and _manifest_is_valid(VAL_MANIFEST)
    )

    if refresh_catalog or rebuild or not manifests_ready:
        entries = ensure_catalog(refresh=refresh_catalog)
        train_entries, val_entries = split_entries(entries)
        save_split_manifests(train_entries, val_entries)
        return train_entries, val_entries

    return (
        load_split_manifest(TRAIN_MANIFEST),
        load_split_manifest(VAL_MANIFEST),
    )


def build_datasets(refresh_catalog: bool = False):
    from training.streaming_dataset import StreamingVideoDataset

    train_entries, val_entries = prepare_training_manifests(
        refresh_catalog=refresh_catalog,
    )

    train_dataset = StreamingVideoDataset(
        train_entries,
        train=True,
        epoch=0,
        max_videos=VIDEOS_PER_EPOCH,
        seed=RANDOM_SEED,
    )

    val_dataset = StreamingVideoDataset(
        val_entries,
        train=False,
        epoch=0,
        max_videos=None,
        seed=RANDOM_SEED,
    )

    return train_dataset, val_dataset
