from backend.services.models.spatial_stub import SpatialStubModel
from backend.services.models.temporal_stub import TemporalStubModel
from backend.services.models.noise_stub import NoiseStubModel


class EnsembleModel:

    def __init__(self):
        self.spatial = SpatialStubModel()
        self.temporal = TemporalStubModel()
        self.noise = NoiseStubModel()

    def predict(self, feature_map):

        spatial_score = self.spatial.predict(
            feature_map.spatial_frames
        )

        temporal_score = self.temporal.predict(
            feature_map.temporal_clips
        )

        noise_score = self.noise.predict(
            feature_map.noise_frames
        )

        final = (
            spatial_score * 0.4 +
            temporal_score * 0.4 +
            noise_score * 0.2
        )

        return {
            "spatial": spatial_score,
            "temporal": temporal_score,
            "noise": noise_score,
            "final": final
        }