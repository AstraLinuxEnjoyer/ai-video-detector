"""Frame-only DAVIS clips: one scene is one sample and one split group."""

import hashlib
from pathlib import Path

from PIL import Image


def collect_davis(base, label):
    records = []
    for images in sorted(base.rglob("JPEGImages")):
        if not images.is_dir() or "DAVIS" not in {p.upper() for p in images.relative_to(base).parts}:
            continue
        resolutions = [images / "Full-Resolution", images / "1080p", images / "480p"]
        selected = next((p for p in resolutions if p.is_dir()), None)
        if selected is None:
            raise ValueError(f"Unsupported DAVIS image layout: {images}")
        for scene in sorted(p for p in selected.iterdir() if p.is_dir()):
            frames = sorted(p.name for p in scene.iterdir()
                            if p.suffix.lower() in {".jpg", ".jpeg"} and p.stem.isdigit())
            if not frames:
                raise ValueError(f"Empty DAVIS scene: {scene}")
            records.append(dict(path=str(scene.resolve()), label=label, source="DAVIS",
                                origin="davis/" + scene.name, kind="image_sequence",
                                relative=scene.relative_to(base).as_posix(), frame_files=frames))
    return records


def audit_sequence(record, hash_file):
    directory = Path(record["path"])
    frames = []
    dimensions = set()
    for name in record["frame_files"]:
        path = directory / name
        stat = path.stat()
        sha = hash_file(path, stat)
        with Image.open(path) as image:
            dimensions.add(image.size)
            image.verify()
        frames.append(dict(name=name, size=stat.st_size, mtime=stat.st_mtime_ns, sha256=sha))
    if len(dimensions) != 1:
        raise ValueError(f"Mixed dimensions in DAVIS scene: {directory}")
    # Hash ordered content, not filenames, to merge copied sequences.
    sha = hashlib.sha256(("image_sequence_v1:" + ":".join(f["sha256"] for f in frames)).encode()).hexdigest()
    width, height = dimensions.pop()
    return dict(record, id=sha, size=sum(f["size"] for f in frames),
                mtime=max(f["mtime"] for f in frames), sequence_frames=frames,
                sequence_width=width, sequence_height=height)


def sequence_unchanged(record):
    directory = Path(record["path"])
    actual = sorted(p.name for p in directory.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg"} and p.stem.isdigit())
    if actual != record["frame_files"]:
        return False
    for item in record["sequence_frames"]:
        stat = (directory / item["name"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime"]):
            return False
    return True
