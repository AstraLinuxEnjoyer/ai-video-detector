"""A fixed native-resolution face region for each motion window."""
import cv2
import numpy as np

from training.spectral.faces import locate

VERSION = 'native-fixed-face-v1'


class FaceUnavailable(ValueError):
    pass


def complete_pairs(records):
    groups = {}
    for row in records:
        groups.setdefault(row['group'], []).append(row)
    return [row for group in groups.values() if sorted(r['label'] for r in group) == [0, 1]
            for row in group]


def crop_face_clip(pixels, size, weights):
    anchors = (0, len(pixels)//2, len(pixels)-1)
    boxes = [locate(cv2.cvtColor(pixels[i], cv2.COLOR_RGB2GRAY), weights) for i in anchors]
    ref = boxes[1]
    if ref is None or all(box is None for box in (boxes[0], boxes[2])):
        raise FaceUnavailable('Face missing in middle or both end frames')
    rx0, ry0, rx1, ry1 = ref
    for box in (boxes[0], boxes[2]):
        if box is None:
            continue
        x0, y0, x1, y1 = box
        if (abs(x0+x1-rx0-rx1) > rx1-rx0 or abs(y0+y1-ry0-ry1) > ry1-ry0
                or not .5 <= (x1-x0)/(rx1-rx0) <= 2 or not .5 <= (y1-y0)/(ry1-ry0) <= 2):
            raise FaceUnavailable('Largest face changes too much across the window')
    valid = [box for box in boxes if box is not None]
    x0, y0 = min(b[0] for b in valid), min(b[1] for b in valid)
    x1, y1 = max(b[2] for b in valid), max(b[3] for b in valid)
    height, width = pixels.shape[1:3]
    side = min(round(1.30*max(x1-x0, y1-y0)), height, width)
    if side < max(x1-x0, y1-y0):
        raise FaceUnavailable('Face region does not fit a fixed square')
    left = max(0, min(width-side, (x0+x1-side)//2))
    top = max(0, min(height-side, (y0+y1-side)//2))
    interpolation = cv2.INTER_AREA if side >= size else cv2.INTER_LINEAR
    return np.stack([cv2.resize(frame[top:top+side, left:left+side], (size, size),
                                interpolation=interpolation) for frame in pixels])
