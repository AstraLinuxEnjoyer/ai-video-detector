"""Label-independent crops and paired degradations, shared by train and inference."""
import io
import random

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]


def load_image(path, max_side=960):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    return image


def degrade(image, rng=None, stress=False):
    """Same distribution for both labels; never infer a transform from the source."""
    rng = rng or random.Random(0)
    scale = .75 if stress else rng.uniform(.65, 1.)
    small = tuple(max(8, round(v * scale)) for v in image.size)
    image = image.resize(small, Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
    if not stress and rng.random() < .2:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.1, .6)))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=65 if stress else rng.randint(55, 95), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def crop_boxes(image, size, rng=None, count=2):
    width, height = image.size
    side = min(size, width, height)
    if rng is None:
        return [(round(x * (width-side)), round(y * (height-side)), side)
                for x, y in ((0, 0), (1, 0), (.5, .5), (0, 1), (1, 1))]
    boxes = []
    for _ in range(count):
        # Prefer textured regions, identically for real/fake, with bounded retries.
        for attempt in range(4):
            x, y = rng.randint(0, width-side), rng.randint(0, height-side)
            box = (x, y, side)
            if np.asarray(image.crop((x, y, x+side, y+side))).std() >= 8 or attempt == 3:
                break
        boxes.append(box)
    return boxes


def patch_tensor(image, size=224, boxes=None):
    boxes = boxes if boxes is not None else crop_boxes(image, size)
    result = []
    for x, y, side in boxes:
        patch = image.crop((x, y, x+side, y+side))
        if patch.size != (size, size):
            patch = patch.resize((size, size), Image.Resampling.BICUBIC)
        result.append(TF.normalize(TF.to_tensor(patch), MEAN, STD))
    return torch.stack(result)


def view_tensor(image, size=224, boxes=None, hybrid=False):
    local = patch_tensor(image, size, boxes)
    if not hybrid:
        return local
    # Full field of view; identical anisotropic resize for both classes, no borders.
    whole = image.resize((size, size), Image.Resampling.BICUBIC)
    global_view = TF.normalize(TF.to_tensor(whole), MEAN, STD)
    return torch.cat([global_view[None], local])


def aggregate_views(scores, hybrid=False):
    return .5*scores[..., 0] + .5*scores[..., 1:].mean(-1) if hybrid else scores.mean(-1)


class PatchFrames(Dataset):
    def __init__(self, records, size=224, train=False, seed=42, stress=False, hybrid=False):
        self.samples = [(r, i, path) for r in records for i, path in enumerate(r["frames"])]
        self.size, self.train, self.seed, self.stress = size, train, seed, stress
        self.hybrid = hybrid
        if train:
            random.Random(seed).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record, frame, path = self.samples[index]
        image = load_image(path)
        if self.train:
            rng = random.Random(f"patch-v1:{self.seed}:{record['id']}:{frame}")
            if rng.random() < .5:
                image = ImageOps.mirror(image)
            boxes = crop_boxes(image, self.size, rng)
            clean = view_tensor(image, self.size, boxes, self.hybrid)
            altered = view_tensor(degrade(image, rng), self.size, boxes, self.hybrid)
            pixels = torch.stack([clean, altered])
        else:
            if self.stress:
                image = degrade(image, stress=True)
            pixels = view_tensor(image, self.size, hybrid=self.hybrid)
        return pixels, float(record["label"]), record["id"], record["source"]
