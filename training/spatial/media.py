"""Bounded CPU decoding before GPU training; originals are never rewritten."""

import io
import json
import math
import random
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .common import atomic_json, event, fingerprint
from .sequences import sequence_unchanged


class SourceChangedError(RuntimeError):
    """A frozen dataset must not silently accept changed source bytes."""


@contextmanager
def decoding_pool(workers):
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        yield executor
    finally:
        # On interruption only in-flight subprocesses finish; queued videos are cancelled.
        executor.shutdown(wait=True, cancel_futures=True)


def probe(path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_streams", "-show_format", "-of", "json", str(path)],
                            capture_output=True, timeout=30, check=True)
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise ValueError("No video stream")
    stream = info["streams"][0]
    duration = float(stream.get("duration") or info["format"].get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid video duration")
    num, den = stream.get("avg_frame_rate", "0/1").split("/")
    fps = float(num) / max(float(den), 1)
    return dict(duration=duration, width=stream["width"], height=stream["height"], fps=fps,
                codec=stream["codec_name"], transfer=stream.get("color_transfer", "unknown"))


def decode(path, timestamp, metadata, size=640):
    filters = []
    if metadata["transfer"] in {"smpte2084", "arib-std-b67"}:
        filters.extend(["zscale=t=linear:npl=100", "format=gbrpf32le",
                        "zscale=p=bt709", "tonemap=tonemap=hable:desat=0",
                        "zscale=t=bt709:m=bt709:r=full"])
    filters.extend([f"scale={size}:{size}:force_original_aspect_ratio=decrease", "format=rgb24"])
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1",
               "-ss", f"{timestamp:.6f}", "-i", str(path), "-an", "-sn", "-dn",
               "-vf", ",".join(filters), "-filter_threads", "1", "-frames:v", "1",
               "-f", "image2pipe", "-vcodec", "png", "-threads", "1", "pipe:1"]
    result = subprocess.run(command, capture_output=True, timeout=45)
    if result.returncode or not result.stdout:
        raise RuntimeError(result.stderr.decode(errors="replace")[-1200:] or "Empty decoded frame")
    with Image.open(io.BytesIO(result.stdout)) as image:
        image.verify()
    return result.stdout


class FrameCache:
    def __init__(self, run, cache_gb, reserve_gb):
        self.root = Path(run) / "frames"
        self.root.mkdir(exist_ok=True)
        self.limit = int(cache_gb * 1024**3)
        self.reserve = int(reserve_gb * 1024**3)
        self.lock = threading.Lock()
        self.used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def put(self, path, content):
        with self.lock:
            previous_size = path.stat().st_size if path.exists() else 0
            if self.used - previous_size + len(content) > self.limit:
                raise OSError("FRAME_CACHE_BUDGET exceeded; increase --cache-gb or reduce video/frame counts")
            if shutil.disk_usage(self.root).free - len(content) < self.reserve:
                raise OSError("DISK_RESERVE reached")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(content)
            tmp.replace(path)
            self.used += len(content) - previous_size

    def clear_train(self, epoch):
        target = (self.root / f"train_{epoch}").resolve()
        if target.parent != self.root.resolve() or not target.name.startswith("train_"):
            raise ValueError("Unexpected cache cleanup path")
        if target.exists():
            shutil.rmtree(target)
        self.used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


def prepare_frames(records, cache, split, count, seed, logger, run, workers=2):
    """Any failed requested video is reported, never replaced with a different one."""
    def one(record):
        path = Path(record["path"])
        sequence = record.get("kind") == "image_sequence"
        if sequence:
            try:
                unchanged = sequence_unchanged(record)
            except OSError:
                unchanged = False
        else:
            stat = path.stat()
            unchanged = stat.st_size == record["size"] and stat.st_mtime_ns == record["mtime"]
        if not unchanged:
            raise SourceChangedError("Original changed since hash audit; start a new run")
        directory = cache.root / split / record["id"]
        info_path = directory / "frames.json"
        signature = fingerprint([record["id"], count, seed, split, record["mtime"]])
        reuse_frames = False
        if info_path.exists():
            saved = json.loads(info_path.read_text())
            reuse_frames = saved["signature"] == signature
            if reuse_frames and all(Path(p).exists() for p in saved["frames"]):
                return dict(record, **saved)
        if sequence:
            metadata = dict(width=record["sequence_width"], height=record["sequence_height"],
                            codec="jpeg_sequence", fps=None, duration=None, frame_count=len(record["frame_files"]))
            end = len(record["frame_files"])
        else:
            metadata = probe(path)
            end = max(0, metadata["duration"] - max(.1, 2 / max(metadata["fps"], 1)))
        rng = random.Random(f"{seed}:{record['id']}:{split}")
        if split.startswith("train_"):
            timestamps = [end * (i + rng.random()) / count for i in range(count)]
        else:
            timestamps = [end * (i + .5) / count for i in range(count)]
        frames = []
        for index, timestamp in enumerate(timestamps):
            output = directory / f"{index:02}.png"
            if not reuse_frames or not output.exists():
                if sequence:
                    frame_index = min(int(timestamp), len(record["frame_files"]) - 1)
                    with Image.open(path / record["frame_files"][frame_index]) as original:
                        image = original.convert("RGB")
                        image.thumbnail((640, 640), Image.Resampling.BICUBIC)
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                    cache.put(output, buffer.getvalue())
                else:
                    cache.put(output, decode(path, timestamp, metadata))
            frames.append(str(output))
        saved = dict(signature=signature, frames=frames, timestamps=None if sequence else timestamps, metadata=metadata)
        if sequence:
            saved["source_frame_indices"] = [min(int(t), len(record["frame_files"])-1) for t in timestamps]
        cache.put(info_path, json.dumps(saved).encode("utf-8"))
        return dict(record, **saved)

    good, failed = [], []
    with decoding_pool(workers) as executor:
        pending = {executor.submit(one, r): r for r in records}
        for index, future in enumerate(tqdm(as_completed(pending), total=len(pending), desc=f"DECODE {split}")):
            record = pending[future]
            try:
                good.append(future.result())
            except Exception as exc:
                detail = dict(stage="decode", split=split, id=record["id"], path=record["path"],
                              source=record["source"], error=str(exc))
                event(run / "data_errors.jsonl", detail)
                failed.append(detail)
                logger.warning("Decode failed [%s]: %s: %s", split, record["path"], exc)
                if isinstance(exc, SourceChangedError) or (
                        isinstance(exc, OSError) and ("BUDGET" in str(exc) or "RESERVE" in str(exc))):
                    for other in pending:
                        other.cancel()
                    raise
            if (index + 1) % 100 == 0:
                logger.info("Decode %s %d/%d successful=%d errors=%d cache=%.2f GiB",
                            split, index + 1, len(records), len(good), len(failed), cache.used / 1024**3)
    good.sort(key=lambda r: r["id"])
    logger.info("Decode %s requested=%d usable=%d failed=%d", split, len(records), len(good), len(failed))
    atomic_json(run / f"{split}_decode.json", dict(requested=len(records), usable=len(good), failures=failed))
    return good, failed
