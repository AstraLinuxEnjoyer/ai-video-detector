import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from training.config import (
    MIN_FREE_DISK_BYTES,
    PREFETCH_MAX_BYTES,
    PREFETCH_MAX_VIDEOS,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
    VIDEO_CACHE_DIR,
)
from training.data.cache import ensure_video_downloaded
from training.data.pipeline import prepare_training_manifests


def _cache_size(path: Path = VIDEO_CACHE_DIR) -> int:
    total = 0
    if not path.exists():
        return total

    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size

    return total


def _has_enough_space() -> bool:
    return shutil.disk_usage(VIDEO_CACHE_DIR).free >= MIN_FREE_DISK_BYTES


def _select_entries(split: str):
    train_entries, val_entries = prepare_training_manifests()

    if split == "train":
        return train_entries
    if split == "val":
        return val_entries

    return [*train_entries, *val_entries]


def prefetch(split: str, max_videos: int, max_gb: float) -> None:
    entries = [
        entry
        for entry in _select_entries(split)
        if entry.local_path is None
    ]

    max_bytes = int(max_gb * 1024**3)
    start_cache_size = _cache_size()
    downloaded = 0

    print()
    print("Prefetch remote videos")
    print(f"Split        : {split}")
    print(f"Max videos   : {max_videos}")
    print(f"Max cache add: {max_gb:.1f} GB")
    print(f"Train manifest: {TRAIN_MANIFEST}")
    print(f"Val manifest  : {VAL_MANIFEST}")
    print()

    for entry in tqdm(entries, desc="PREFETCH"):
        if downloaded >= max_videos:
            break

        if _cache_size() - start_cache_size >= max_bytes:
            print("Prefetch stopped: cache budget reached.")
            break

        if not _has_enough_space():
            print("Prefetch stopped: free disk space is below safety limit.")
            break

        ensure_video_downloaded(entry)
        downloaded += 1

    added = (_cache_size() - start_cache_size) / 1024**3

    print()
    print(f"Downloaded videos: {downloaded}")
    print(f"Cache added      : {added:.2f} GB")
    print(f"Cache directory  : {VIDEO_CACHE_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "val", "all"],
        default="train",
        help="Which manifest split to prefetch.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=PREFETCH_MAX_VIDEOS,
        help="Maximum number of remote videos to download.",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=PREFETCH_MAX_BYTES / 1024**3,
        help="Maximum cache growth in GB.",
    )
    args = parser.parse_args()

    prefetch(
        split=args.split,
        max_videos=args.max_videos,
        max_gb=args.max_gb,
    )


if __name__ == "__main__":
    main()
