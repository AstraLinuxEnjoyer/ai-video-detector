import time
import random

from backend.services.models.ensemble import EnsembleModel
from backend.services import (
    task_service,
    video_processor,
    feature_extractor
)



class VideoWorker:

    def run(self):
        print("Worker started...")

        while True:

            tasks = task_service.get_queued_tasks()

            if not tasks:
                time.sleep(2)
                continue

            for task in tasks:
                self.process(task.task_id)

    def process(self, task_id: str):

        print(f"Processing task: {task_id}")

        task_service.mark_processing(task_id)

        frames = video_processor.extract_frames(
            task_id=task_id,
            fps=1
        )

        feature_map = feature_extractor.build(
            task_id,
            frames
        )

        ensemble = EnsembleModel()

        result = ensemble.predict(feature_map)

        print("SPATIAL:", result["spatial"])
        print("TEMPORAL:", result["temporal"])
        print("NOISE:", result["noise"])
        print("FINAL:", result["final"])

        for progress in [10, 30, 50, 70, 90, 100]:
            time.sleep(1)
            task_service.update_progress(task_id, progress)

        task_service.update_result(
            task_id,
            result["final"]
        )

        print(f"Done: {task_id}")