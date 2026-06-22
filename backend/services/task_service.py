import json
import os
from datetime import datetime

from backend.schemas.task import TaskSchema
from backend.services.storage_service import StorageService


class TaskService:

    def __init__(self):
        self.storage_service = StorageService()

    def create(self, task_id: str, filename: str):

        task = TaskSchema(
            task_id=task_id,
            status="QUEUED",
            progress=0,
            filename=filename,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            result=None,
            error=None
        )

        self.save(task)

        return task

    def save(self, task: TaskSchema):

        path = self.storage_service.get_task_path(task.task_id)

        metadata = path / "metadata" / "task.json"

        with open(metadata, "w", encoding="utf-8") as f:
            f.write(task.model_dump_json(indent=4))

    def load(self, task_id: str):

        path = self.storage_service.get_task_path(task_id)

        metadata = path / "metadata" / "task.json"

        with open(metadata, "r", encoding="utf-8") as f:
            data = json.load(f)

        return TaskSchema(**data)

    def update_status(self, task_id: str, status: str):

        task = self.load(task_id)

        task.status = status
        task.updated_at = datetime.now()

        self.save(task)

    def update_progress(self, task_id: str, progress: int):

        task = self.load(task_id)

        task.progress = progress
        task.updated_at = datetime.now()

        self.save(task)

    def update_result(self, task_id: str, probability: float):

        task = self.load(task_id)

        task.result = probability
        task.progress = 100
        task.status = "COMPLETED"
        task.updated_at = datetime.now()

        self.save(task)

    def get_task_path(self, task_id: str):
        return self.storage_service.get_task_path(task_id)

    def get_all_tasks(self):
        base = self.storage_service.base_path  # папка uploads
        return [p.name for p in base.iterdir() if p.is_dir()]


    def get_task(self, task_id: str):
        return self.load(task_id)


    def get_queued_tasks(self):
        tasks = []

        for task_id in self.get_all_tasks():
            try:
                task = self.load(task_id)
                if task.status == "QUEUED":
                    tasks.append(task)
            except Exception:
                continue

        return tasks

    def mark_processing(self, task_id: str):
        task = self.load(task_id)

        task.status = "PROCESSING"
        task.updated_at = datetime.now()

        self.save(task)