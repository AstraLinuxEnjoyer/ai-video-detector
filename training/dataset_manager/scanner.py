from dataclasses import dataclass
from pathlib import Path

from training.dataset_manager.config import (
    REAL_DIR,
    FAKE_DIR,
    VIDEO_EXTENSIONS,
)


@dataclass
class VideoInfo:

    path: Path

    label: int

    label_name: str

    source: str

    filename: str


class DatasetScanner:

    def __init__(self):

        self.videos = []

    def scan(self):

        self.videos.clear()

        self._scan_directory(
            REAL_DIR,
            0,
            "real"
        )

        self._scan_directory(
            FAKE_DIR,
            1,
            "fake"
        )

        print()

        print("=" * 60)

        print("Dataset Scanner")

        print("=" * 60)

        print(f"Total videos : {len(self.videos)}")

        print(
            f"Real videos  : "
            f"{len([v for v in self.videos if v.label == 0])}"
        )

        print(
            f"Fake videos  : "
            f"{len([v for v in self.videos if v.label == 1])}"
        )

        print("=" * 60)

        return self.videos

    def _scan_directory(
        self,
        root: Path,
        label: int,
        label_name: str,
    ):

        if not root.exists():

            print(f"Directory not found: {root}")

            return

        for file in root.rglob("*"):

            if not file.is_file():

                continue

            if file.suffix.lower() not in VIDEO_EXTENSIONS:

                continue

            relative = file.relative_to(root)

            source = relative.parts[0]

            self.videos.append(

                VideoInfo(

                    path=file,

                    label=label,

                    label_name=label_name,

                    source=source,

                    filename=file.name

                )

            )


if __name__ == "__main__":

    scanner = DatasetScanner()

    videos = scanner.scan()

    print()

    print("First 10 videos:")

    print()

    for video in videos[:10]:

        print(video)