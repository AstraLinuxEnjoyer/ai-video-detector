"""Pinned OpenCV YuNet for locating a native-resolution face region, not classifying it."""

import hashlib
import tempfile
import threading
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from filelock import FileLock

from training.spatial.common import ROOT, replace_file

NAME = "face_detection_yunet_2026may.onnx"
SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
SIZE = 229738
URL = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/{NAME}"
MODEL = ROOT / "cache/spectral/models" / NAME
INPUT = 640
_THREAD = threading.local()


def ensure_weights(path=MODEL):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=60):
        if path.exists():
            if path.stat().st_size == SIZE and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256:
                return path
            raise ValueError(f"YuNet weights checksum mismatch: {path}")
        with urlopen(URL, timeout=45) as response:
            blob = response.read(SIZE + 1)
        if len(blob) != SIZE or hashlib.sha256(blob).hexdigest() != SHA256:
            raise ValueError("Downloaded YuNet weights do not match the pinned OpenCV Zoo release")
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            try:
                stream.write(blob)
            finally:
                stream.close()
        try:
            replace_file(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path


def native_box(detections, width, height, scaled_width, scaled_height):
    if detections is None:
        return None
    candidates = []
    for detection in detections:
        x, y, w, h = (float(value) for value in detection[:4])
        if not np.isfinite(detection).all() or w <= 0 or h <= 0:
            continue
        if not (0 <= x+w/2 < scaled_width and 0 <= y+h/2 < scaled_height):
            continue
        sx, sy = width/scaled_width, height/scaled_height
        # Retain some context at the face boundary without resizing the selected pixels.
        x0, y0 = max(0, int((x-.10*w)*sx)), max(0, int((y-.10*h)*sy))
        x1, y1 = min(width, round((x+1.10*w)*sx)), min(height, round((y+1.10*h)*sy))
        if min(x1-x0, y1-y0) >= 16:
            candidates.append(((x1-x0)*(y1-y0), (x0, y0, x1, y1)))
    return max(candidates)[1] if candidates else None


def locate(gray, weights):
    height, width = gray.shape
    key = str(weights)
    if getattr(_THREAD, "key", None) != key:
        _THREAD.detector = cv2.FaceDetectorYN.create(key, "", (INPUT, INPUT), .75, .3, 1000)
        _THREAD.key = key
    scale = min(1, INPUT/max(width, height))
    scaled_width, scaled_height = max(1, round(width*scale)), max(1, round(height*scale))
    small = cv2.resize(gray, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((INPUT, INPUT, 3), dtype=np.uint8)
    canvas[:scaled_height, :scaled_width] = small[:, :, None]
    _, detected = _THREAD.detector.detect(canvas)
    return native_box(detected, width, height, scaled_width, scaled_height)
