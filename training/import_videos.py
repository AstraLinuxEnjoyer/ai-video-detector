import argparse
import os
import random
import shutil
from pathlib import Path

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


def import_video(
    source_path: Path,
    output_path: Path,
    mode: str,
    replace: bool,
    strict_hardlink: bool,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if not replace:
            return "exists"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Downloaded dataset folder.")
    parser.add_argument("--label", choices=["real", "fake"], required=True)
    parser.add_argument("--source", required=True, help="Source folder name.")
    parser.add_argument("--limit", type=int, default=None)
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
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_root = LOCAL_REAL_DIR if args.label == "real" else LOCAL_FAKE_DIR
    output_dir = output_root / safe_name(args.source)

    videos = find_videos(input_dir)
    rng = random.Random(args.seed)
    rng.shuffle(videos)

    if args.limit is not None:
        videos = videos[: max(0, args.limit)]

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
