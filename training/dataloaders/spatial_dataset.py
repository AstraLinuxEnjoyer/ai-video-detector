from pathlib import Path

from PIL import Image

from torch.utils.data import Dataset


class SpatialDataset(Dataset):

    def __init__(self, root, transform=None):

        self.transform = transform

        self.samples = []

        root = Path(root)

        classes = {
            "real": 0,
            "fake": 1
        }

        for class_name, label in classes.items():

            class_dir = root / class_name

            if not class_dir.exists():
                continue

            for video_dir in class_dir.iterdir():

                if not video_dir.is_dir():
                    continue

                for image in video_dir.glob("*.jpg"):

                    self.samples.append(
                        (
                            image,
                            label
                        )
                    )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label