from pathlib import Path
import random

import cv2
import torch

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms


IMAGE_SIZE = 384


class SpatialDataset(Dataset):

    def __init__(self, samples, train=True):

        self.samples = samples

        self.train = train

        if train:

            self.transform = transforms.Compose([

                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

                transforms.RandomHorizontalFlip(),

                transforms.RandomRotation(10),

                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),

                transforms.ToTensor(),

                transforms.Normalize(

                    mean=[0.485, 0.456, 0.406],

                    std=[0.229, 0.224, 0.225]

                )

            ])

        else:

            self.transform = transforms.Compose([

                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

                transforms.ToTensor(),

                transforms.Normalize(

                    mean=[0.485, 0.456, 0.406],

                    std=[0.229, 0.224, 0.225]

                )

            ])

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        video_path, label = self.samples[index]

        frame = self.load_random_frame(video_path)

        image = Image.fromarray(frame)

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

    def load_random_frame(self, video_path):

        cap = cv2.VideoCapture(str(video_path))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 1:

            frame_number = 0

        else:

            frame_number = random.randint(0, frame_count - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = cap.read()

        cap.release()

        if not success:

            raise RuntimeError(f"Cannot read frame from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame