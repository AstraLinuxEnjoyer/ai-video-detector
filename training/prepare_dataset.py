"""
Builds train/val manifests from the remote DeepAction v1 dataset.

Videos are NOT downloaded here — only metadata is fetched from HuggingFace.
Actual video files are downloaded on demand during training.
"""

import argparse
from collections import Counter

from training.config import (
    MANUAL_DATASET_CANDIDATES,
    REMOTE_DATASETS,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
)
from training.data.pipeline import prepare_training_manifests


def _print_sources() -> None:
    print("Automatic remote sources:")
    for source in REMOTE_DATASETS:
        if not source.get("enabled", True):
            continue
        print(f"  - {source['name']}: {source['repo_id']}")

    print()
    print("Manual candidates, not auto-downloaded:")
    for source in MANUAL_DATASET_CANDIDATES:
        print(f"  - {source}")


def _print_split_stats(name: str, entries) -> None:
    labels = Counter(entry.label for entry in entries)
    datasets = Counter(entry.dataset for entry in entries)
    sources = Counter(entry.source for entry in entries)

    print(
        f"{name} videos : {len(entries)} "
        f"(real={labels[0]}, fake={labels[1]})"
    )
    for dataset, count in sorted(datasets.items()):
        print(f"  {dataset}: {count}")
    print("  sources:")
    for source, count in sorted(sources.items()):
        print(f"    {source}: {count}")


def main(refresh_catalog: bool = False, rebuild: bool = False):
    print("\nRemote dataset preparation\n")
    _print_sources()
    print()

    train_entries, val_entries = prepare_training_manifests(
        refresh_catalog=refresh_catalog,
        rebuild=rebuild,
    )

    _print_split_stats("Train", train_entries)
    _print_split_stats("Val", val_entries)
    print(f"Train manifest: {TRAIN_MANIFEST}")
    print(f"Val manifest  : {VAL_MANIFEST}")
    print("\nManifests ready. Optional prefetch:")
    print("  python -m training.prefetch_datasets --split train --max-gb 20")
    print("\nStart training with:")
    print("  python -m training.train_spatial --fresh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh the remote catalog and rebuild train/val manifests.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild train/val manifests from cached and local videos.",
    )
    args = parser.parse_args()

    main(refresh_catalog=args.refresh, rebuild=args.rebuild)
