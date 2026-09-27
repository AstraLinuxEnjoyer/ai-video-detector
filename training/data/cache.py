from pathlib import Path
import shutil

from filelock import FileLock
from huggingface_hub import hf_hub_download

from training.config import MIN_FREE_DISK_BYTES, MIN_VIDEO_BYTES, ROOT, VIDEO_CACHE_DIR
from training.data.catalog import VideoEntry


def _local_video_path(entry: VideoEntry) -> Path:
    return _repo_cache_dir(entry) / entry.repo_path


def _repo_cache_dir(entry: VideoEntry) -> Path:
    if entry.repo_id:
        return VIDEO_CACHE_DIR / entry.repo_id.replace("/", "__")

    return VIDEO_CACHE_DIR


def _legacy_video_path(entry: VideoEntry) -> Path:
    return VIDEO_CACHE_DIR / entry.repo_path


def _lock_path(entry: VideoEntry) -> Path:
    return VIDEO_CACHE_DIR / ".locks" / f"{entry.id}.lock"


def _ensure_free_disk_space() -> None:
    usage = shutil.disk_usage(VIDEO_CACHE_DIR)
    if usage.free < MIN_FREE_DISK_BYTES:
        gb_free = usage.free / 1024**3
        gb_required = MIN_FREE_DISK_BYTES / 1024**3
        raise RuntimeError(
            "Not enough free disk space for video cache: "
            f"{gb_free:.1f} GB free, need at least {gb_required:.1f} GB."
        )


def ensure_video_downloaded(entry: VideoEntry) -> Path:
    if entry.local_path is not None:
        local_path = Path(entry.local_path)
        if not local_path.is_absolute():
            local_path = ROOT / local_path
        if local_path.exists() and local_path.stat().st_size >= MIN_VIDEO_BYTES:
            return local_path
        raise FileNotFoundError(f"Local video not found: {local_path}")

    target = _local_video_path(entry)

    if target.exists() and target.stat().st_size >= MIN_VIDEO_BYTES:
        return target

    legacy_target = _legacy_video_path(entry)
    if legacy_target.exists() and legacy_target.stat().st_size >= MIN_VIDEO_BYTES:
        return legacy_target

    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_path(entry)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path), timeout=600):
        if target.exists() and target.stat().st_size >= MIN_VIDEO_BYTES:
            return target

        _ensure_free_disk_space()

        if entry.repo_id is None:
            raise RuntimeError(f"Remote video entry has no repo_id: {entry.id}")

        downloaded = hf_hub_download(
            repo_id=entry.repo_id,
            repo_type=entry.repo_type,
            filename=entry.repo_path,
            local_dir=str(_repo_cache_dir(entry)),
        )

        downloaded_path = Path(downloaded)
        if downloaded_path.exists():
            return downloaded_path

        if target.exists():
            return target

    raise RuntimeError(f"Failed to download {entry.repo_path}")
