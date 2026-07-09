import json
import random

from pathlib import Path

from PIL import Image

import torch

from torch.utils.data import Dataset

from torchvision import transforms

from training.config import (

    IMAGE_SIZE,

    FRAMES_PER_VIDEO,

    VIDEOS_PER_EPOCH,

)


class VideoFrameDataset(Dataset):

    def __init__(

        self,

        manifest_path,

        train=True,

    ):

        with open(

            manifest_path,

            encoding="utf-8"

        ) as f:

            videos = json.load(f)

        if train:

            random.shuffle(videos)

            videos = videos[:VIDEOS_PER_EPOCH]

        self.samples = []

        for video in videos:

            frames_dir = Path(video["frames"])

            label = video["label"]

            if not frames_dir.exists():

                continue

            frames = list(

                frames_dir.glob("*.jpg")

            )

            if len(frames) == 0:

                continue

            count = min(

                FRAMES_PER_VIDEO,

                len(frames)

            )

            selected = random.sample(

                frames,

                count

            )

            for frame in selected:

                self.samples.append(

                    (

                        frame,

                        label,

                    )

                )

        random.shuffle(

            self.samples

        )

        self.transform = transforms.Compose([

            transforms.Resize(

                (

                    IMAGE_SIZE,

                    IMAGE_SIZE,

                )

            ),

            transforms.ToTensor(),

            transforms.Normalize(

                [0.485,0.456,0.406],

                [0.229,0.224,0.225]

            )

        ])

        print(

            f"Loaded frames: {len(self.samples)}"

        )

    def __len__(self):

        return len(

            self.samples

        )

    def __getitem__(

        self,

        index,

    ):

        image_path, label = self.samples[index]

        image = Image.open(

            image_path

        ).convert("RGB")

        image = self.transform(image)

        return (

            image,

            torch.tensor(

                label,

                dtype=torch.float32

            )

        )