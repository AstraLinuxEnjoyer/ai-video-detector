import argparse
import os
import random
import shutil
from pathlib import Path

from training.dataset_paths import GENVIDEO_ARCHIVE_SOURCES, archive_source_path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_REAL_DIR = ROOT / "data" / "real"
LOCAL_FAKE_DIR = ROOT / "data" / "fake"
RANDOM_SEED = 42
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MIN_VIDEO_BYTES = 1024


def find_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and path.stat().st_size >= MIN_VIDEO_BYTES
        )
    )


def safe_name(value: str) -> str:
    result = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            result.append(char)
        else:
            result.append("_")
    return "".join(result)


def same_volume(source_path: Path, output_path: Path) -> bool:
    parent = output_path.parent
    while not parent.exists():
        if parent.parent == parent:
            raise FileNotFoundError(f"Destination volume unavailable: {output_path}")
        parent = parent.parent
    return source_path.stat().st_dev == parent.stat().st_dev


def import_video(
    source_path: Path,
    output_path: Path,
    mode: str,
    replace: bool,
    strict_hardlink: bool,
) -> str:
    if output_path.exists():
        if not replace:
            return "exists"

    # Cross-volume hardlinks cannot work. Never silently refill the system disk.
    if mode == "hardlink" and not same_volume(source_path, output_path):
        raise OSError(
            "Source and destination are on different volumes; hardlinks are unavailable. "
            "Use --mode copy explicitly (consumes space at the destination), "
            "or move the working data directory and reconnect it first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    if mode == "copy":
        shutil.copy2(source_path, output_path)
        return "copied"

    try:
        os.link(source_path, output_path)
        return "linked"
    except OSError:
        if strict_hardlink:
            raise
        shutil.copy2(source_path, output_path)
        return "copied"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="Downloaded dataset folder.")
    inputs.add_argument("--dataset", choices=sorted(GENVIDEO_ARCHIVE_SOURCES),
                        help="Source under the configured archive root (datasets/archive).")
    parser.add_argument("--dataset-root", type=Path,
                        help="Override archive root, e.g. D:/dataset; used with --dataset.")
    parser.add_argument("--label", choices=["real", "fake"], required=True)
    parser.add_argument("--source", required=True, help="Source folder name.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="List the plan without writing files.")
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="Hardlink saves disk space when source and repo are on the same drive.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing imported files.",
    )
    parser.add_argument(
        "--strict-hardlink",
        action="store_true",
        help="Fail instead of copying if hardlink creation is denied.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    if args.dataset_root is not None and not args.dataset:
        parser.error("--dataset-root requires --dataset")
    if args.dataset and (args.label != "fake" or args.source != args.dataset):
        parser.error("Named GenVideo sources require --label fake and the matching --source")

    input_dir = archive_source_path(args.dataset, args.dataset_root) if args.dataset else Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_root = LOCAL_REAL_DIR if args.label == "real" else LOCAL_FAKE_DIR
    output_dir = output_root / safe_name(args.source)

    videos = find_videos(input_dir)
    rng = random.Random(args.seed)
    rng.shuffle(videos)

    if args.limit is not None:
        videos = videos[: max(0, args.limit)]

    if args.dry_run:
        existing = sum((output_dir / f"{index:06d}_{safe_name(path.stem)}{path.suffix.lower()}").exists()
                       for index, path in enumerate(videos, start=1))
        print(f"Input : {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Planned: {len(videos)}; existing: {existing}; mode: {args.mode}")
        if videos and args.mode == "hardlink" and not same_volume(videos[0], output_dir / "probe.mp4"):
            print("Different volumes: missing files cannot be hardlinked; no copy or replacement performed.")
        return

    counts = {
        "linked": 0,
        "copied": 0,
        "exists": 0,
    }

    for index, video_path in enumerate(videos, start=1):
        suffix = video_path.suffix.lower()
        name = f"{index:06d}_{safe_name(video_path.stem)}{suffix}"
        output_path = output_dir / name
        status = import_video(
            video_path,
            output_path,
            args.mode,
            replace=args.replace,
            strict_hardlink=args.strict_hardlink,
        )
        counts[status] += 1

    print()
    print("Import complete")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Label : {args.label}")
    print(f"Source: {args.source}")
    print(f"Total : {len(videos)}")
    print(f"Linked: {counts['linked']}")
    print(f"Copied: {counts['copied']}")
    print(f"Exists: {counts['exists']}")
    print()
    print("Next:")
    print("  python -m training.prepare_dataset --rebuild")


if __name__ == "__main__":
    main()
