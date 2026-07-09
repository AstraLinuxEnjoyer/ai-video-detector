import argparse
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
from tqdm import tqdm

from training.config import (
    LOCAL_FAKE_DIR,
    LOCAL_REAL_DIR,
    MIN_VIDEO_BYTES,
    VIDEO_EXTENSIONS,
)

try:
    cv2.setLogLevel(0)
except Exception:
    pass


def find_videos(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in VIDEO_EXTENSIONS
            and path.stat().st_size >= MIN_VIDEO_BYTES
        )
    )


def can_read_video(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False

    success, frame = cap.read()
    cap.release()

    return bool(success and frame is not None)


def transcode_to_h264(input_path: Path, output_path: Path, crf: int) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not available in PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-an",
        str(output_path),
    ]

    result = subprocess.run(command)
    return (
        result.returncode == 0
        and output_path.exists()
        and output_path.stat().st_size >= MIN_VIDEO_BYTES
        and can_read_video(output_path)
    )


def repair_video(path: Path, crf: int, dry_run: bool) -> str:
    if can_read_video(path):
        return "ok"

    if dry_run:
        return "needs_repair"

    temp_path = path.with_suffix(path.suffix + ".repair_tmp.mp4")

    try:
        if not transcode_to_h264(path, temp_path, crf):
            if temp_path.exists():
                temp_path.unlink()
            return "failed"

        path.unlink()
        temp_path.replace(path)
        return "repaired"

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        return "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=None,
        help="Folder to scan. Defaults to data/real and data/fake.",
    )
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    roots = [Path(args.root)] if args.root else [LOCAL_REAL_DIR, LOCAL_FAKE_DIR]

    videos: list[Path] = []
    for root in roots:
        if root.exists():
            videos.extend(find_videos(root))

    counts = {
        "ok": 0,
        "needs_repair": 0,
        "repaired": 0,
        "failed": 0,
    }

    failed: list[Path] = []

    for path in tqdm(videos, desc="REPAIR"):
        status = repair_video(
            path,
            crf=args.crf,
            dry_run=args.dry_run,
        )
        counts[status] += 1
        if status == "failed":
            failed.append(path)

    print()
    print("Video repair summary")
    print(f"Scanned      : {len(videos)}")
    print(f"OK           : {counts['ok']}")
    print(f"Needs repair : {counts['needs_repair']}")
    print(f"Repaired     : {counts['repaired']}")
    print(f"Failed       : {counts['failed']}")

    if failed:
        print()
        print("Failed files:")
        for path in failed[:100]:
            print(f"  {path}")

    print()
    print("Next:")
    print("  python -m training.prepare_dataset --rebuild")


if __name__ == "__main__":
    main()
