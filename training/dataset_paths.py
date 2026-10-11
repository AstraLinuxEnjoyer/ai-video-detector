"""Downloaded archives are separate from the frozen working training data."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DATASET_DIR = Path(os.environ.get(
    "AI_VIDEO_ARCHIVE_ROOT", str(ROOT / "datasets" / "archive")
)).expanduser()

GENVIDEO_ARCHIVE_SOURCES = {
    "GenVideo_OpenSora": "train_OpenSora",
    "GenVideo_pika": "train_pika",
    "GenVideo_SVD": "train_SVD",
    "GenVideo_VideoCrafter": "train_VideoCrafter",
    "GenVideo_ZeroScope": "train_ZeroScope",
}


def archive_source_path(source, root=None):
    if source not in GENVIDEO_ARCHIVE_SOURCES:
        raise ValueError(f"Unknown archive source: {source}")
    base = Path(root) if root is not None else ARCHIVE_DATASET_DIR
    path = base / "GenVideo-100K" / "extracted" / GENVIDEO_ARCHIVE_SOURCES[source]
    if not path.is_dir():
        raise FileNotFoundError(
            f"Archive source missing: {path}. Check datasets/archive, "
            "AI_VIDEO_ARCHIVE_ROOT or --dataset-root."
        )
    return path.resolve()
