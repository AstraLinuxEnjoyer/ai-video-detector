import random


class SpatialStubModel:

    def predict(self, frames):
        """
        frames: list of Path objects
        """

        if not frames:
            return 0.0

        # имитация "понимания кадров"
        base = random.uniform(0.1, 0.9)

        # чуть зависит от количества кадров
        noise = min(len(frames) / 300, 0.1)

        return min(base + noise, 1.0)