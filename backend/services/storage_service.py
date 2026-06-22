from pathlib import Path
from backend.core.config import settings
from backend.utils.id_generator import generate_task_id


class StorageService:
    """
    Файловое хранилище задач анализа видео
    """

    def __init__(self):
        self.base_path = Path(settings.UPLOAD_DIR)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_task(self) -> str:
        task_id = generate_task_id()

        task_path = self._get_task_path(task_id)
        task_path.mkdir(parents=True, exist_ok=True)

        (task_path / "frames").mkdir(exist_ok=True)
        (task_path / "logs").mkdir(exist_ok=True)
        (task_path / "metadata").mkdir(exist_ok=True)

        return task_id

    def save_video(self, task_id: str, file_bytes: bytes, filename: str = "input.mp4") -> Path:
        task_path = self._get_task_path(task_id)
        video_path = task_path / filename

        with open(video_path, "wb") as f:
            f.write(file_bytes)

        return video_path

    def get_task_path(self, task_id: str) -> Path:
        return self._get_task_path(task_id)

    def get_frames_path(self, task_id: str) -> Path:
        return self._get_task_path(task_id) / "frames"

    def get_logs_path(self, task_id: str) -> Path:
        return self._get_task_path(task_id) / "logs"

    def _get_task_path(self, task_id: str) -> Path:
        return self.base_path / task_id