from pathlib import Path
import subprocess

from backend.services.storage_service import StorageService


class VideoProcessor:

    def __init__(self):
        self.storage_service = StorageService()

    def extract_frames(
        self,
        task_id: str,
        fps: int = 1
    ):

        task_path = self.storage_service.get_task_path(task_id)

        input_video = task_path / "input.mp4"

        frames_dir = task_path / "frames"

        frames_dir.mkdir(exist_ok=True)

        output_pattern = frames_dir / "frame_%05d.jpg"

        command = [
            "ffmpeg",
            "-i",
            str(input_video),
            "-vf",
            f"fps={fps}",
            str(output_pattern),
            "-y"
        ]

        result = subprocess.run(command)

        if result.returncode != 0:
            raise RuntimeError("FFmpeg extraction failed")

        return sorted(frames_dir.glob("*.jpg"))