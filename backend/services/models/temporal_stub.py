import random


class TemporalStubModel:

    def predict(self, clips):
        """
        clips: list of [frames]
        """

        if not clips:
            return 0.0

        base = random.uniform(0.1, 0.9)

        # если много клипов → сложнее видео → выше шанс fake
        factor = min(len(clips) / 50, 0.2)

        return min(base + factor, 1.0)