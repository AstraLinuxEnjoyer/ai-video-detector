import random
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from training.config import (
    FRAME_PAD_COLOR,
    FRAMES_PER_VIDEO,
    IMAGE_SIZE,
    PRESERVE_ASPECT_RATIO,
    VAL_FRAMES_PER_VIDEO,
)
from training.data.cache import ensure_video_downloaded
from training.data.catalog import VideoEntry

try:
    cv2.setLogLevel(0)
except Exception:
    pass


def _frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return max(frame_count, 1)


def _decode_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if PRESERVE_ASPECT_RATIO:
        height, width = frame.shape[:2]
        scale = image_size / max(height, width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))

        frame = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        canvas = np.full(
            (image_size, image_size, 3),
            FRAME_PAD_COLOR,
            dtype=np.uint8,
        )
        y = (image_size - resized_height) // 2
        x = (image_size - resized_width) // 2
        canvas[y:y + resized_height, x:x + resized_width] = frame
        return canvas

    frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)

    return frame


def _decode_png_bytes(raw: bytes, image_size: int) -> np.ndarray:
    array = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError("ffmpeg returned an unreadable frame")

    return _decode_frame(frame, image_size)


def _ffprobe_duration(video_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None

    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None

    if duration <= 0:
        return None

    return duration


def _read_frame_ffmpeg(
    video_path: Path,
    image_size: int,
    timestamp: float | None = None,
) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not available for fallback decoding")

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    if timestamp is not None and timestamp > 0:
        command.extend(["-ss", f"{timestamp:.3f}"])

    command.extend([
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ])

    result = subprocess.run(
        command,
        capture_output=True,
    )

    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(
            "ffmpeg fallback failed: "
            f"{result.stderr.decode(errors='ignore').strip()}"
        )

    return _decode_png_bytes(result.stdout, image_size)


def _read_frame(video_path: Path, frame_number: int, image_size: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return _read_frame_ffmpeg(video_path, image_size)

    frame_count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_number = max(0, min(frame_number, frame_count - 1))

    candidates = [frame_number]
    for offset in range(1, min(8, frame_count)):
        previous_frame = frame_number - offset
        next_frame = frame_number + offset

        if previous_frame >= 0:
            candidates.append(previous_frame)
        if next_frame < frame_count:
            candidates.append(next_frame)

    for candidate in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        success, frame = cap.read()

        if success and frame is not None:
            cap.release()
            return _decode_frame(frame, image_size)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        success, frame = cap.read()

        if success and frame is not None:
            cap.release()
            return _decode_frame(frame, image_size)

        position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if position >= frame_count:
            break

    cap.release()

    timestamp = frame_number / fps if fps > 0 else None
    return _read_frame_ffmpeg(video_path, image_size, timestamp=timestamp)


def load_random_frame(video_path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    try:
        frame_count = _frame_count(video_path)
    except RuntimeError:
        duration = _ffprobe_duration(video_path)
        timestamp = random.uniform(0, max(duration, 0.001)) if duration else None
        return _read_frame_ffmpeg(video_path, image_size, timestamp=timestamp)

    if frame_count <= 1:
        frame_number = 0
    else:
        frame_number = random.randint(0, frame_count - 1)

    return _read_frame(video_path, frame_number, image_size)


def load_fixed_frame(
    video_path: Path,
    sample_index: int,
    samples_per_video: int,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    try:
        frame_count = _frame_count(video_path)
    except RuntimeError:
        duration = _ffprobe_duration(video_path)

        if duration and samples_per_video > 1:
            timestamp = sample_index * duration / (samples_per_video - 1)
        else:
            timestamp = None

        return _read_frame_ffmpeg(video_path, image_size, timestamp=timestamp)

    if frame_count <= 1 or samples_per_video <= 1:
        frame_number = 0
    else:
        frame_number = round(
            sample_index * (frame_count - 1) / (samples_per_video - 1)
        )

    return _read_frame(video_path, frame_number, image_size)


class StreamingVideoDataset(Dataset):

    def __init__(
        self,
        entries: list[VideoEntry],
        train: bool = True,
        epoch: int = 0,
        max_videos: int | None = None,
        seed: int = 42,
    ):
        self.entries = entries
        self.train = train
        self.epoch = epoch
        self.max_videos = max_videos
        self.seed = seed
        self.samples: list[tuple[VideoEntry, int]] = []

        self.transform = self._build_transform(train)
        self._rebuild_samples()

    def _build_transform(self, train: bool):
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._rebuild_samples()

    def _balanced_epoch_videos(
        self,
        videos: list[VideoEntry],
        rng: random.Random,
    ) -> list[VideoEntry]:
        if not self.train or self.max_videos is None or len(videos) <= self.max_videos:
            result = list(videos)
            rng.shuffle(result)
            return result

        per_class = max(1, self.max_videos // 2)
        real = [entry for entry in videos if entry.is_real]
        fake = [entry for entry in videos if not entry.is_real]

        selected = [
            *self._source_round_robin(real, per_class, rng),
            *self._source_round_robin(fake, per_class, rng),
        ]

        if len(selected) < self.max_videos:
            selected_ids = {entry.id for entry in selected}
            remaining = [
                entry
                for entry in videos
                if entry.id not in selected_ids
            ]
            rng.shuffle(remaining)
            selected.extend(remaining[: self.max_videos - len(selected)])

        rng.shuffle(selected)
        return selected[: self.max_videos]

    def _source_round_robin(
        self,
        videos: list[VideoEntry],
        count: int,
        rng: random.Random,
    ) -> list[VideoEntry]:
        by_source: dict[str, list[VideoEntry]] = {}
        for entry in videos:
            by_source.setdefault(entry.source, []).append(entry)

        queues = []
        for source_entries in by_source.values():
            source_entries = list(source_entries)
            rng.shuffle(source_entries)
            queues.append(source_entries)

        rng.shuffle(queues)
        selected: list[VideoEntry] = []

        while len(selected) < count and queues:
            next_queues = []
            for queue in queues:
                if len(selected) >= count:
                    break
                if queue:
                    selected.append(queue.pop())
                if queue:
                    next_queues.append(queue)

            queues = next_queues
            rng.shuffle(queues)

        return selected

    def _rebuild_samples(self) -> None:
        rng = random.Random(self.seed + self.epoch)

        videos = list(self.entries)
        videos = self._balanced_epoch_videos(videos, rng)

        frames_per_video = FRAMES_PER_VIDEO if self.train else VAL_FRAMES_PER_VIDEO

        samples: list[tuple[VideoEntry, int]] = []
        for entry in videos:
            for frame_idx in range(frames_per_video):
                samples.append((entry, frame_idx))

        rng.shuffle(samples)
        self.samples = samples

        split_name = "train" if self.train else "val"
        print(
            f"[{split_name}] epoch={self.epoch}: "
            f"{len(videos)} videos -> {len(self.samples)} frame samples"
        )

        if self.train:
            sources = Counter(entry.source for entry in videos)
            source_summary = ", ".join(
                f"{source}={count}"
                for source, count in sorted(sources.items())
            )
            print(f"[{split_name}] sources: {source_summary}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, index: int):
        entry, frame_idx = self.samples[index]

        video_path = ensure_video_downloaded(entry)

        if self.train:
            frame = load_random_frame(video_path)
        else:
            frame = load_fixed_frame(
                video_path,
                sample_index=frame_idx,
                samples_per_video=VAL_FRAMES_PER_VIDEO,
            )

        image = Image.fromarray(frame)
        image = self.transform(image)

        label = torch.tensor(entry.label, dtype=torch.float32)

        return image, label, entry.id

    def __getitem__(self, index: int):
        last_error = None
        max_attempts = min(10, len(self.samples))

        for offset in range(max_attempts):
            sample_index = (index + offset) % len(self.samples)
            try:
                return self._load_sample(sample_index)
            except Exception as exc:
                last_error = exc
                if offset == 0:
                    entry, _frame_idx = self.samples[sample_index]
                    print(f"[skip] {entry.source}: {entry.repo_path} ({exc})")

        raise RuntimeError(
            f"Failed to load {max_attempts} consecutive samples. "
            f"Last error: {last_error}"
        )
