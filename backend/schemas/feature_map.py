from pydantic import BaseModel


class FeatureMap(BaseModel):

    spatial_frames: list[str]

    noise_frames: list[str]

    temporal_clips: list[list[str]]