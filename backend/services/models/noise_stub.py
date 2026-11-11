import random


class NoiseStubModel:

    def predict(self, frames):

        if not frames:
            return 0.0

        base = random.uniform(0.1, 0.9)

        # шум зависит от количества кадров (условная метрика)
        factor = min(len(frames) / 500, 0.15)

        return min(base + factor, 1.0)