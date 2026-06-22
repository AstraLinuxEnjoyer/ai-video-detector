import json

from backend.schemas.feature_map import FeatureMap
from backend.services import StorageService

storage_service = StorageService()

class FeatureExtractor:

    def build(self, task_id: str, frames):

        spatial = self._build_spatial(frames)

        temporal = self._build_temporal(frames)

        noise = spatial

        feature_map = FeatureMap(

            spatial_frames=[
                f.name for f in spatial
            ],

            noise_frames=[
                f.name for f in noise
            ],

            temporal_clips=[

                [
                    frame.name
                    for frame in clip
                ]

                for clip in temporal

            ]

        )

        metadata_path = (
            storage_service
            .get_task_path(task_id)
            / "metadata"
        )

        metadata_path.mkdir(exist_ok=True)

        with open(
            metadata_path / "feature_map.json",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                feature_map.model_dump_json(
                    indent=4
                )
            )

        return feature_map

    def _build_spatial(self, frames):

        total = len(frames)

        if total <= 256:

            return frames

        step = total / 256

        result = []

        for i in range(256):

            result.append(
                frames[int(i * step)]
            )

        return result

    def _build_temporal(
        self,
        frames,
        clip_length=16,
        clips_count=32
    ):

        total = len(frames)

        if total <= clip_length:

            return [frames]

        clips_count = min(
            clips_count,
            total // clip_length
        )

        max_start = total - clip_length

        step = max_start / clips_count

        clips = []

        for i in range(clips_count):

            start = int(i * step)

            clips.append(
                frames[
                    start:
                    start + clip_length
                ]
            )

        return clips