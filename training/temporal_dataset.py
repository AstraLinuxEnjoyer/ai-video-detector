import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from training.data.cache import ensure_video_downloaded
from training.data.catalog import VideoEntry
from training.streaming_dataset import (
    _ffprobe_duration,
    _frame_count,
    _read_frame,
    _read_frame_ffmpeg,
)
from training.temporal_config import (
    RANDOM_SEED,
    TEMPORAL_CLIP_LENGTH,
    TEMPORAL_CLIPS_PER_VIDEO,
    TEMPORAL_FRAME_STRIDE,
    TEMPORAL_IMAGE_SIZE,
    TEMPORAL_VAL_CLIPS_PER_VIDEO,
)


class TemporalVideoDataset(Dataset):
    def __init__(
        self,
        entries: list[VideoEntry],
        train: bool = True,
        epoch: int = 0,
        max_videos: int | None = None,
        seed: int = RANDOM_SEED,
    ):
        self.entries = entries
        self.train = train
        self.epoch = epoch
        self.max_videos = max_videos
        self.seed = seed
        self.samples: list[tuple[VideoEntry, int]] = []
        self.video_count = 0

        self._rebuild_samples()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._rebuild_samples()

    def source_counts(self) -> Counter:
        return Counter(entry.source for entry, _clip_idx in self.samples)

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

    def _balanced_videos(
        self,
        videos: list[VideoEntry],
        rng: random.Random,
    ) -> list[VideoEntry]:
        if self.max_videos is None or len(videos) <= self.max_videos:
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

    def _rebuild_samples(self) -> None:
        rng = random.Random(self.seed + self.epoch)

        videos = self._balanced_videos(list(self.entries), rng)
        clips_per_video = (
            TEMPORAL_CLIPS_PER_VIDEO
            if self.train
            else TEMPORAL_VAL_CLIPS_PER_VIDEO
        )

        samples: list[tuple[VideoEntry, int]] = []
        for entry in videos:
            for clip_idx in range(clips_per_video):
                samples.append((entry, clip_idx))

        rng.shuffle(samples)

        self.samples = samples
        self.video_count = len(videos)

        split_name = "train" if self.train else "val"
        print(
            f"[temporal:{split_name}] epoch={self.epoch}: "
            f"{len(videos)} videos -> {len(samples)} clip samples"
        )

        sources = Counter(entry.source for entry in videos)
        source_summary = ", ".join(
            f"{source}={count}"
            for source, count in sorted(sources.items())
        )
        print(f"[temporal:{split_name}] sources: {source_summary}")

    def __len__(self) -> int:
        return len(self.samples)

    def _clip_frame_numbers(
        self,
        frame_count: int,
        clip_idx: int,
        rng: random.Random,
    ) -> list[int]:
        if frame_count <= 1:
            return [0] * TEMPORAL_CLIP_LENGTH

        if frame_count <= TEMPORAL_CLIP_LENGTH:
            return [
                round(i * (frame_count - 1) / max(TEMPORAL_CLIP_LENGTH - 1, 1))
                for i in range(TEMPORAL_CLIP_LENGTH)
            ]

        stride = max(1, TEMPORAL_FRAME_STRIDE)
        span = (TEMPORAL_CLIP_LENGTH - 1) * stride + 1

        if span >= frame_count:
            stride = max(1, frame_count // TEMPORAL_CLIP_LENGTH)
            span = (TEMPORAL_CLIP_LENGTH - 1) * stride + 1

        max_start = max(0, frame_count - span)

        if self.train:
            start = rng.randint(0, max_start) if max_start > 0 else 0
        else:
            clips_per_video = max(TEMPORAL_VAL_CLIPS_PER_VIDEO, 1)
            if clips_per_video <= 1:
                start = max_start // 2
            else:
                start = round(clip_idx * max_start / (clips_per_video - 1))

        return [
            min(start + i * stride, frame_count - 1)
            for i in range(TEMPORAL_CLIP_LENGTH)
        ]

    def _clip_timestamps(
        self,
        duration: float,
        clip_idx: int,
        rng: random.Random,
    ) -> list[float]:
        duration = max(duration, 0.001)
        clip_duration = min(duration, 2.0)
        max_start = max(duration - clip_duration, 0.0)

        if self.train:
            start = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
        else:
            clips_per_video = max(TEMPORAL_VAL_CLIPS_PER_VIDEO, 1)
            if clips_per_video <= 1:
                start = max_start / 2
            else:
                start = clip_idx * max_start / (clips_per_video - 1)

        if TEMPORAL_CLIP_LENGTH <= 1:
            return [start]

        return [
            min(start + i * clip_duration / (TEMPORAL_CLIP_LENGTH - 1), duration)
            for i in range(TEMPORAL_CLIP_LENGTH)
        ]

    def _load_clip_frames(
        self,
        video_path: Path,
        clip_idx: int,
        rng: random.Random,
    ):
        try:
            frame_count = _frame_count(video_path)
            frame_numbers = self._clip_frame_numbers(
                frame_count,
                clip_idx,
                rng,
            )
            return [
                _read_frame(video_path, frame_number, TEMPORAL_IMAGE_SIZE)
                for frame_number in frame_numbers
            ]
        except Exception:
            duration = _ffprobe_duration(video_path)
            if duration is None:
                raise

            timestamps = self._clip_timestamps(duration, clip_idx, rng)
            return [
                _read_frame_ffmpeg(
                    video_path,
                    TEMPORAL_IMAGE_SIZE,
                    timestamp=timestamp,
                )
                for timestamp in timestamps
            ]

    def _to_tensor(self, frames, rng: random.Random) -> torch.Tensor:
        tensor = torch.from_numpy(np.stack(frames)).float()
        tensor = tensor / 255.0

        if self.train and rng.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])

        # T,H,W,C -> C,T,H,W for torchvision video models.
        tensor = tensor.permute(3, 0, 1, 2).contiguous()

        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)

        return (tensor - mean) / std

    def _load_sample(self, index: int):
        entry, clip_idx = self.samples[index]
        video_path = ensure_video_downloaded(entry)

        rng = random.Random(
            self.seed
            + self.epoch * 1_000_003
            + index * 97
        )
        frames = self._load_clip_frames(video_path, clip_idx, rng)
        clip = self._to_tensor(frames, rng)
        label = torch.tensor(entry.label, dtype=torch.float32)

        return clip, label, entry.id

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
                    entry, _clip_idx = self.samples[sample_index]
                    print(f"[temporal:skip] {entry.source}: {entry.repo_path} ({exc})")

        raise RuntimeError(
            f"Failed to load {max_attempts} consecutive temporal samples. "
            f"Last error: {last_error}"
        )
